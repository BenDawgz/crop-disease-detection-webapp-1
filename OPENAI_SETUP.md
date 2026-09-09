# Run CropScan with OpenAI on Windows

The existing frontend now uses the OpenAI Responses API by default, with
`gpt-5.4-mini`. This is a starting model, not a validated plant-disease detector.
Compare its results against expert-labelled field photos before relying on it.

1. Create an API key in your OpenAI API account and enable API billing.
2. Open `instance/openai_api_key.txt` in Notepad. Replace the placeholder with
   your API key only, then save. Do not paste the key into chat or frontend code.
   This private file is excluded from Git. Alternatively set `OPENAI_API_KEY`
   in the environment used to start Flask; that takes priority.
3. Stop the running app with Ctrl+C in its terminal, then restart from PowerShell:

   ```powershell
   cd C:\Users\Doaltech\Desktop\cddw
   .\.venv\Scripts\python.exe app.py
   ```

4. Open http://127.0.0.1:5000 and upload a photo. Internet access is required.
   API usage is billed separately from a ChatGPT subscription.

No new Python dependency is needed in the existing environment. New installations
can use `pip install -r requirements-windows.txt` in their virtual environment.

## Behaviour

- Photos are resized to at most 1600 pixels per side, stripped of metadata by
  re-encoding, and sent to OpenAI. The upload page discloses this.
- Results include crop, visible symptoms, possible causes, and next steps.
  No invented confidence percentage or automatic disease-specific treatment.
- Non-leaf, unsupported crop, uncertain, unreadable, and service-error results
  remain distinct. API errors never silently fall back to the local model.
- Successful results are cached in `instance/vision_cache` by image content,
  model and assessment instructions. Refreshing does not repeat a successful
  request. Failed requests can be retried. This request lock is for a single
  desktop server process; a multi-worker deployment needs shared locking.
- `store: false` disables Responses application-state storage; this is not a
  promise of zero provider retention. Original uploads and assessment logs
  remain on this PC under the app's existing storage behaviour.
- Local retraining is disabled while OpenAI is selected. Provisional API
  assessments are stored in recent-search history with 0 as an unavailable
  score, not a measured probability.

To choose another compatible image/structured-output model, set `OPENAI_MODEL`
before starting. Access depends on your API account.

To return to the preserved refined local model:

```powershell
$env:ANALYSIS_PROVIDER = 'local'
.\.venv\Scripts\python.exe app.py
```

For OpenAI again, set `$env:ANALYSIS_PROVIDER = 'openai'` and restart.
Keep this desktop server on localhost; public deployment also needs user access
controls and request limits to protect paid usage.

Documentation: [Image inputs](https://developers.openai.com/api/docs/guides/images-vision),
[Structured outputs](https://developers.openai.com/api/docs/guides/structured-outputs).
