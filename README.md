# Insta Agent

Chat UI + agent that sends Instagram DMs. LLM: Groq `openai/gpt-oss-120b`, fallback Gemini.

## Run
```
pip install -r requirements.txt
cp .env.example .env     # add keys
uvicorn main:app --reload
```
Open http://localhost:8000. Keep `DRY_RUN=true` until you've tested, then set `false`.

## Instagram modes
**graph (official, safe):** needs an Instagram Business/Creator account + Meta app with messaging permission.
Point the webhook to `https://<public-url>/webhook` (use ngrok), verify token = `IG_VERIFY_TOKEN`.
Official API can only message people who DMed you in the last 24h; the webhook builds the contact list.

**instagrapi (unofficial):** `pip install instagrapi`, set `INSTAGRAM_MODE=instagrapi` + IG_USERNAME/IG_PASSWORD.
Can DM any username, but Instagram may flag/ban the account. Use a throwaway account.
