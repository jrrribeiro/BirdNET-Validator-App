import gradio as gr
from typing import Protocol

from src.cache.ephemeral_cache_manager import EphemeralCacheManager
from src.domain.models import Detection
from src.repositories.in_memory_detection_repository import InMemoryDetectionRepository
from src.services.audio_fetch_service import AudioFetchService
from src.services.detection_queue_service import DetectionQueueService


class _AudioFetchResultProtocol(Protocol):
    cache_key: str
    local_path: str
    source: str


class _AudioServiceProtocol(Protocol):
    def fetch(self, dataset_repo: str, audio_id: str) -> _AudioFetchResultProtocol: ...

    def cleanup_after_validation(self, cache_key: str) -> None: ...


def _seed_service() -> DetectionQueueService:
    repo = InMemoryDetectionRepository()
    demo_items = [
        Detection(
            detection_key="0000000000001001",
            audio_id="audio_1001",
            scientific_name="Cyanocorax cyanopogon",
            confidence=0.93,
            start_time=1.2,
            end_time=2.5,
        ),
        Detection(
            detection_key="0000000000001002",
            audio_id="audio_1002",
            scientific_name="Ramphastos toco",
            confidence=0.88,
            start_time=0.8,
            end_time=2.1,
        ),
        Detection(
            detection_key="0000000000001003",
            audio_id="audio_1003",
            scientific_name="Cyanocorax cyanopogon",
            confidence=0.72,
            start_time=3.1,
            end_time=4.0,
        ),
        Detection(
            detection_key="0000000000001004",
            audio_id="audio_1004",
            scientific_name="Psarocolius decumanus",
            confidence=0.67,
            start_time=5.0,
            end_time=6.3,
        ),
    ]
    repo.seed("demo-project", demo_items)
    return DetectionQueueService(repo)


def _page_to_table(service: DetectionQueueService, page: int, scientific_name: str, min_confidence: float):
    filter_name = scientific_name.strip() if scientific_name.strip() else None
    page_obj = service.get_page(
        project_slug="demo-project",
        page=page,
        page_size=2,
        scientific_name=filter_name,
        min_confidence=min_confidence,
    )

    rows = [
        [
            item.detection_key,
            item.audio_id,
            item.scientific_name,
            round(item.confidence, 3),
            item.start_time,
            item.end_time,
        ]
        for item in page_obj.items
    ]
    status = f"Pagina {page_obj.page}/{page_obj.total_pages} | Total filtrado: {page_obj.total_items}"
    return rows, status, page_obj.page


def _extract_audio_id(rows: object, selected_index: int) -> str:
    normalized_rows: list[list[object]]

    if hasattr(rows, "values"):
        normalized_rows = [list(item) for item in rows.values.tolist()]
    else:
        normalized_rows = [list(item) for item in rows] if rows else []

    if not normalized_rows:
        raise ValueError("Nenhuma deteccao carregada na tabela")
    if selected_index < 0 or selected_index >= len(normalized_rows):
        raise ValueError("Selecione uma deteccao valida na tabela")

    value = normalized_rows[selected_index][1]
    audio_id = str(value).strip()
    if not audio_id:
        raise ValueError("audio_id invalido na deteccao selecionada")
    return audio_id


def _fetch_selected_audio(
    audio_service: _AudioServiceProtocol,
    dataset_repo: str,
    rows: object,
    selected_index: int,
    previous_cache_key: str,
) -> tuple[str | None, str, str]:
    repo = dataset_repo.strip()
    if not repo:
        return None, "", "Informe dataset repo no formato owner/repo"

    try:
        audio_id = _extract_audio_id(rows=rows, selected_index=selected_index)
        result = audio_service.fetch(dataset_repo=repo, audio_id=audio_id)
        status = f"Audio carregado ({result.source}) para audio_id={audio_id}"
        return result.local_path, result.cache_key, status
    except Exception as exc:
        if previous_cache_key:
            return None, previous_cache_key, f"Falha ao carregar audio: {exc}"
        return None, "", f"Falha ao carregar audio: {exc}"


def _cleanup_selected_audio(audio_service: _AudioServiceProtocol, cache_key: str) -> tuple[str, str | None]:
    if not cache_key:
        return "Nenhum audio em cache para limpar", None

    audio_service.cleanup_after_validation(cache_key=cache_key)
    return "Cache de audio limpo apos validacao", None


def create_app() -> gr.Blocks:
    service = _seed_service()
    audio_service = AudioFetchService(EphemeralCacheManager(ttl_seconds=300, max_files=128))

    with gr.Blocks(title="BirdNET Validator HF") as demo:
        gr.Markdown("# BirdNET Validator HF")
        gr.Markdown("Sprint 2: fila paginada + audio sob demanda com cache efemero.")

        dataset_repo = gr.Textbox(label="Dataset repo", value="SEU_USUARIO/birdnet-projeto-dataset")

        with gr.Row():
            species_filter = gr.Textbox(label="Filtro especie", placeholder="Ex: Cyanocorax cyanopogon")
            min_confidence = gr.Slider(label="Confianca minima", minimum=0.0, maximum=1.0, step=0.01, value=0.0)

        with gr.Row():
            prev_btn = gr.Button("Pagina anterior")
            next_btn = gr.Button("Proxima pagina")
            refresh_btn = gr.Button("Aplicar filtros")

        page_state = gr.State(value=1)
        table = gr.Dataframe(
            headers=["detection_key", "audio_id", "scientific_name", "confidence", "start_time", "end_time"],
            label="Deteccoes",
            interactive=False,
        )
        selected_index = gr.Number(label="Linha selecionada", value=0, precision=0)

        with gr.Row():
            load_audio_btn = gr.Button("Carregar audio selecionado")
            clear_audio_btn = gr.Button("Limpar cache apos validacao")

        audio_player = gr.Audio(label="Audio sob demanda", type="filepath")
        cache_key_state = gr.State(value="")
        status = gr.Textbox(label="Status", interactive=False)

        def refresh(page: int, species: str, confidence: float):
            return _page_to_table(service, page=page, scientific_name=species, min_confidence=confidence)

        def go_next(page: int, species: str, confidence: float):
            return refresh(page + 1, species, confidence)

        def go_prev(page: int, species: str, confidence: float):
            return refresh(max(1, page - 1), species, confidence)

        def on_select(evt):
            if isinstance(evt.index, tuple):
                return int(evt.index[0])
            if isinstance(evt.index, int):
                return int(evt.index)
            return 0

        demo.load(
            fn=refresh,
            inputs=[page_state, species_filter, min_confidence],
            outputs=[table, status, page_state],
        )
        refresh_btn.click(
            fn=lambda species, confidence: refresh(1, species, confidence),
            inputs=[species_filter, min_confidence],
            outputs=[table, status, page_state],
        )
        next_btn.click(
            fn=go_next,
            inputs=[page_state, species_filter, min_confidence],
            outputs=[table, status, page_state],
        )
        prev_btn.click(
            fn=go_prev,
            inputs=[page_state, species_filter, min_confidence],
            outputs=[table, status, page_state],
        )
        table.select(fn=on_select, inputs=None, outputs=[selected_index])
        load_audio_btn.click(
            fn=lambda repo, rows, idx, cache_key: _fetch_selected_audio(
                audio_service=audio_service,
                dataset_repo=repo,
                rows=rows,
                selected_index=int(idx),
                previous_cache_key=cache_key,
            ),
            inputs=[dataset_repo, table, selected_index, cache_key_state],
            outputs=[audio_player, cache_key_state, status],
        )
        clear_audio_btn.click(
            fn=lambda cache_key: _cleanup_selected_audio(audio_service=audio_service, cache_key=cache_key),
            inputs=[cache_key_state],
            outputs=[status, audio_player],
        )

    return demo
