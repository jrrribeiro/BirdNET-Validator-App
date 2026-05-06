# BirdNET Validator App - Testing Guide for HF Space

This guide explains how to test the BirdNET Validator application on Hugging Face Spaces.

## Quick Overview

The validator is a Gradio application for human review of BirdNET detections. It provides:
- Multi-project login with role-based access
- Detection queue with filtering and pagination
- Audio playback and validation actions
- Concurrency-safe writes with optimistic locking
- Conflict resolution and reporting

## Architecture

```
┌─────────────────────────────────────────────────┐
│          Gradio Web UI (app.py)                 │
├─────────────────────────────────────────────────┤
│  ├─ Login Page       (auth_service)             │
│  ├─ Admin Panel      (admin_panel)              │
│  ├─ Queue Display    (detection_queue_service)  │
│  └─ Audio Player     (audio_fetch_service)      │
├─────────────────────────────────────────────────┤
│  Services & Repositories                        │
│  ├─ ValidationService    (save/validate/report) │
│  ├─ DetectionQueueService (filter/paginate)     │
│  ├─ AudioFetchService    (download/cache)       │
│  └─ AppendOnlyValidationRepository (atomic ops) │
├─────────────────────────────────────────────────┤
│  Data Sources                                   │
│  ├─ Projects (bootstrap/config)                 │
│  ├─ Users & Access Control                      │
│  ├─ Detections (HF dataset or seed file)        │
│  └─ Validations (local append-only log)         │
└─────────────────────────────────────────────────┘
```

## Testing on HF Spaces

### Step 1: Configure HF Space with Demo Data

1. Go to **BirdNET-Validator-App Space Settings**
2. Configure Variables/Secrets:

```
BIRDNET_ENABLE_DEMO_BOOTSTRAP=true
BIRDNET_PAGE_SIZE=25
```

These settings will:
- Load built-in demo users: `admin_user`, `demo_user`, `validator_demo`
- Load demo project: `demo-project`
- Load demo detections: 100 sample birds

### Step 2: Access the App

1. Go to your HF Space URL
2. Wait for the Gradio interface to load
3. You should see the login page

### Step 3: Test Login

**Test user 1 (Admin):**
- Username: `admin_user`
- Expected: Access to all projects with admin role
- Expected UI: "Admin Panel" tab visible

**Test user 2 (Validator):**
- Username: `demo_user`
- Expected: Access to demo-project with validator role
- Expected UI: "Admin Panel" tab NOT visible

**Test user 3 (Another Validator):**
- Username: `validator_demo`
- Expected: Same access as demo_user

### Step 4: Test Core Features

#### Feature 1: Project Selection

1. Login as `admin_user`
2. Click "Select Project"
3. Verify: `demo-project` appears in list
4. Click to select `demo-project`
5. Expected: Queue loads with 100 detections

#### Feature 2: Detection Queue & Filtering

1. Verify detections display with:
   - Audio ID
   - Scientific Name
   - Confidence score
   - Current validation status

2. Test filters:
   - **Filter by Species**: Enter scientific name prefix, queue updates
   - **Filter by Confidence**: Set min/max bounds, queue filters
   - **Search**: Type audio ID, rows filtered in real-time

3. Test pagination:
   - Default page size: 25 items
   - Click "Next Page" / "Previous Page"
   - Verify correct items load

#### Feature 3: Audio Playback

1. Click on a detection row
2. Click "Play Selected" button
3. Expected: Audio player loads with sample WAV
4. Verify audio plays correctly
5. Verify sample rate and duration display

#### Feature 4: Validation Actions

1. Select a detection by clicking the row
2. Choose validation status:
   - **✓ Positive**: Species confirmed
   - **✗ Negative**: False alarm
   - **? Uncertain**: Unsure
   - **⊘ Skip**: Come back later
3. Optional: Add notes in the text field
4. Click "Save Validation"
5. Expected UI changes:
   - Row highlights in green (valid) or yellow (uncertain)
   - Status displays in "Validation Status" column
   - Validator name appears in "Validated By" column

#### Feature 5: Batch Validation (Conflicts)

1. Simulate conflict: Validate same detection from 2 different users
   - In separate browser windows:
     - Window 1: Login as `admin_user`
     - Window 2: Login as `demo_user`
   - Both validate the same detection differently
2. Window 2 should show: "Optimistic lock conflict"
3. Click "Reapply my validation" to override
4. Expected: Validation saved, conflict row marked

#### Feature 6: Admin Panel (Admin Only)

1. Login as `admin_user`
2. Click "Admin Panel" tab
3. Verify options:
   - View validation report
   - Create invites
   - Manage user roles
   - View audit logs

4. Click "Validation Report"
5. Expected: Shows statistics:
   - Total validations
   - Breakdown by validator
   - Breakdown by status
   - Last updated time

## Keyboard Shortcuts (if enabled)

Press '?' in the app to see shortcuts:
- `P`: Positive validation
- `N`: Negative validation
- `U`: Uncertain validation
- `S`: Skip detection
- `→`: Next page
- `←`: Previous page
- `?`: Show this help

## Data Flow During Test

1. **User logs in**
   - Auth service validates credentials
   - RuntimeConfig loads projects and user access
   - Dashboard loads project list

2. **User selects project**
   - DetectionQueueService loads first page (25 items)
   - Each detection loads from in-memory repository

3. **User clicks on detection**
   - AudioFetchService fetches from HF dataset
   - Audio cached in ephemeral memory
   - Audio player displays WAV

4. **User validates**
   - ValidationService creates validation event
   - AppendOnlyValidationRepository appends to log
   - Optimistic locking prevents conflicts
   - UI updates with new status

## Expected Test Results

✅ **All should pass:**
- [ ] Login with `admin_user` succeeds
- [ ] Login with `demo_user` succeeds
- [ ] Project selection loads demo-project
- [ ] Queue displays 100 detections
- [ ] Filters reduce queue size
- [ ] Audio playback works
- [ ] Positive/Negative/Uncertain/Skip actions work
- [ ] Validation status updates in real-time
- [ ] Pagination works (next/prev page)
- [ ] Notes are saved with validation
- [ ] Admin panel visible for `admin_user`
- [ ] Validation report shows correct stats
- [ ] Multiple concurrent validations don't lose data
- [ ] Conflict resolution works

## Troubleshooting

### "Loading..." spinner stuck

**Cause**: Audio fetch service timeout or missing audio

**Solution**:
1. Check HF dataset is accessible
2. Verify `BIRDNET_DETECTIONS_FILE` is set correctly
3. Check browser console for errors

### "Optimistic lock conflict" on first validation

**Cause**: Concurrent writes detected (expected in conflict test)

**Solution**:
1. Click "Reapply my validation" to override
2. Or click "Discard" to cancel

### "Project not found"

**Cause**: Bootstrap didn't load projects

**Solution**:
1. Ensure `BIRDNET_ENABLE_DEMO_BOOTSTRAP=true`
2. Or provide `BIRDNET_PROJECTS_FILE` pointing to valid JSON
3. Check Space logs for errors

### Audio player blank

**Cause**: Missing audio or fetch service error

**Solution**:
1. Verify HF dataset has audio files
2. Check `BIRDNET_DETECTIONS_FILE` points to valid detection JSON
3. Verify audio file paths in detection data

## Local Testing Before HF

To test locally before deploying to HF:

```bash
# Test with demo bootstrap
export BIRDNET_ENABLE_DEMO_BOOTSTRAP=true
export BIRDNET_PAGE_SIZE=25
python app.py

# Open browser to http://localhost:7860
```

Then follow the testing steps above.

## Performance Expectations

- **Login**: < 1 second
- **Project load**: < 2 seconds
- **Queue page load**: < 1 second
- **Audio fetch**: 2-5 seconds (depends on file size)
- **Validation save**: < 1 second
- **Pagination**: < 1 second

## Next Steps After Testing

1. ✅ Confirm all features work locally
2. ✅ Push to HF Space
3. ✅ Test on HF Space URL
4. ✅ Configure production data (real projects/users/detections)
5. ✅ Set up invite email (optional, requires EmailJS)
6. ✅ Share HF Space URL with validators

## Files Reference

**Main entry point:**
- `app.py` - Starts Gradio server

**UI Components:**
- `src/ui/app_factory.py` - Creates Gradio interface
- `src/ui/login_page.py` - Login form
- `src/ui/admin_panel.py` - Admin panel UI

**Services:**
- `src/services/validation_service.py` - Validation logic
- `src/services/detection_queue_service.py` - Queue filtering/pagination
- `src/services/audio_fetch_service.py` - Audio download/caching
- `src/auth/auth_service.py` - Authentication

**Data:**
- `src/repositories/append_only_validation_repository.py` - Write validation events
- `src/repositories/in_memory_detection_repository.py` - Read detections
- `src/config/runtime_config.py` - Configuration from env vars
- `src/cache/ephemeral_cache_manager.py` - Temporary audio cache

## Security Notes

- Tokens are validated against Hugging Face API
- Each user needs `read` access to the detection dataset
- Each user needs `write` access to create validations
- Admin users can manage invites and ACLs
- All writes are append-only (immutable audit log)

## Questions or Issues?

Check:
1. HF Space logs (Settings > Logs)
2. Browser console (F12 > Console)
3. README.md for configuration details
4. app_factory.py for initialization code
