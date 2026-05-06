# 🐦 BirdNET Validator - Quick Test Guide

## 📋 Sumário Executivo

Você tem 2 opções para testar o validator:

### **Opção 1: Teste Local (5 minutos)**
```bash
export BIRDNET_ENABLE_DEMO_BOOTSTRAP=true
python app.py
# Abrir http://localhost:7860
```

### **Opção 2: Teste no HF Space (10 minutos)**
1. Vá para: https://huggingface.co/spaces/jrrribeiro/BirdNET-Validator-App
2. Clique em "Settings" → "Variables"
3. Adicione: `BIRDNET_ENABLE_DEMO_BOOTSTRAP=true`
4. Salve e aguarde o app restart
5. Teste as funcionalidades abaixo

---

## 🚀 Quick Test (5 minutos)

### Passo 1: Login
**Teste com 3 usuários diferentes:**

| Usuário | Função | Esperado |
|---------|--------|----------|
| `admin_user` | Admin | ✓ Admin Panel visível |
| `demo_user` | Validator | ✓ Sem Admin Panel |
| `validator_demo` | Validator | ✓ Sem Admin Panel |

**Para testar:**
1. Coloca o username no campo
2. Clica "Login"
3. Verifica se conseguiu

---

### Passo 2: Seleciona Projeto
1. Clica em "Select Project"
2. Esperado: `demo-project` aparece
3. Clica em `demo-project`
4. Aguarda carregar (carregamento mostra 100 detections)

---

### Passo 3: Validação Simples
1. Clica em qualquer linha da tabela
2. Clica "Play Selected" para ouvir
3. Escolhe uma ação:
   - ✓ **Positive** (sim, é o pássaro)
   - ✗ **Negative** (não é o pássaro)
   - ? **Uncertain** (não tenho certeza)
   - ⊘ **Skip** (pula para depois)
4. Clica "Save Validation"
5. **Esperado:** Linha muda de cor (verde=valid, amarelo=uncertain)

---

### Passo 4: Filtros
1. No campo "Filter by Species": digita `Zoothera` (qualquer nome científico)
2. **Esperado:** Tabela filtra apenas essa espécie
3. Limpa o filtro, tabela volta a mostrar tudo

---

### Passo 5: Admin Panel (apenas admin_user)
1. Login como `admin_user`
2. Clica em "Admin Panel" tab
3. Clica "Validation Report"
4. **Esperado:** Estatísticas aparecem (total de validações, por usuário, etc.)

---

## 📊 Full Feature Checklist

Marque ✓ à medida que testa:

### Login & Authorization
- [ ] `admin_user` login succeed
- [ ] `demo_user` login succeed
- [ ] `validator_demo` login succeed
- [ ] Admin panel only visible for admin_user

### Project Management
- [ ] Project selection loads demo-project
- [ ] Queue loads 100 detections
- [ ] Switching projects reloads queue

### Queue Display & Filtering
- [ ] Detections show: Audio ID, Species, Confidence, Status
- [ ] Filter by Species works
- [ ] Filter by Confidence works
- [ ] Pagination works (next/prev page)
- [ ] Page size = 25 items

### Audio Playback
- [ ] Click row → audio player loads
- [ ] Click "Play Selected" → audio plays
- [ ] Audio duration displays correctly

### Validation Actions
- [ ] Can mark as Positive ✓
- [ ] Can mark as Negative ✗
- [ ] Can mark as Uncertain ?
- [ ] Can mark as Skip ⊘
- [ ] Can add validation notes
- [ ] Row color changes after validation
- [ ] Status updates in real-time

### Conflict Handling (Advanced)
- [ ] Open 2 browser windows
- [ ] Window 1: Login as admin_user
- [ ] Window 2: Login as demo_user
- [ ] Window 1: Validate detection as "Positive"
- [ ] Window 2: Validate SAME detection as "Negative"
- [ ] Window 2 shows: "Optimistic lock conflict" message
- [ ] Can click "Reapply" to override

### Admin Features (admin_user only)
- [ ] Admin Panel tab visible
- [ ] Can view Validation Report
- [ ] Report shows correct statistics
- [ ] Can invite new users (if enabled)

### Data Persistence
- [ ] Validations saved after refresh
- [ ] Can resume validation from where you left off
- [ ] Validation history preserved

---

## 🎯 Expected Results Summary

| Feature | Status | Notes |
|---------|--------|-------|
| Login | ✓ | Demo users work |
| Queue | ✓ | 100 demo detections |
| Audio | ✓ | Sample WAV plays |
| Validation | ✓ | All 4 actions work |
| Filters | ✓ | Species & confidence |
| Pagination | ✓ | 25 items/page |
| Conflicts | ✓ | Optimistic lock works |
| Admin Panel | ✓ | For admin_user only |
| Reporting | ✓ | Statistics display |

---

## ⚠️ Troubleshooting

### "App Loading..." stuck
→ Check HF Space logs: Settings → Logs
→ Verify `BIRDNET_ENABLE_DEMO_BOOTSTRAP=true` set

### "Login fails"
→ Use exact username: `admin_user` (with underscore)
→ Don't include `.hf` domain

### "Audio player blank"
→ Try different detection (some may have missing audio in demo)
→ Check browser console for errors (F12)

### "Validation doesn't save"
→ Check validation status column updates
→ Refresh page to see if it persisted
→ Check HF Space logs for errors

---

## 📞 Commands to Test Locally

```bash
# 1. Verify pre-deployment checks
python scripts/check_deployment.py

# 2. Run unit tests
pytest tests/unit/ -q

# 3. Start app with demo bootstrap
export BIRDNET_ENABLE_DEMO_BOOTSTRAP=true
python app.py

# 4. Test with specific port
export PORT=7860
python app.py
```

---

## 🎬 Test Video Script (if needed)

1. **0:00-0:30** - Login as admin_user
2. **0:30-0:45** - Select project
3. **0:45-1:15** - Show queue with 25 detections
4. **1:15-1:45** - Play audio sample
5. **1:45-2:15** - Validate: Positive, Negative, Uncertain
6. **2:15-2:45** - Show filters (species, confidence)
7. **2:45-3:00** - Show Admin Panel report

---

## 📚 Documentation

- **Setup**: See `README.md`
- **Architecture**: See `HF_SPACE_TESTING.md` (detailed)
- **Troubleshooting**: See `HF_SPACE_TESTING.md` (detailed)
- **Code**: See `src/ui/app_factory.py` (main app)

---

## ✅ Sign-Off

When all features work:

```
Date: ____________
Tester: __________
Environment: HF Space / Local
Status: ✓ READY FOR PRODUCTION
```

---

**Questions? Check:**
- HF Space Logs (Settings > Logs)
- Browser Console (F12)
- `HF_SPACE_TESTING.md` (detailed guide)
- `app_factory.py` (source code)

**Ready to test? Go to:**
→ https://huggingface.co/spaces/jrrribeiro/BirdNET-Validator-App
