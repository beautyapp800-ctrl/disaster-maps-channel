import json, sys, webbrowser
sys.path.insert(0, "C:/disaster-maps")
from pathlib import Path
from google_auth_oauthlib.flow import InstalledAppFlow
from automation import youtube
cs = sys.argv[1]
flow = InstalledAppFlow.from_client_secrets_file(cs, youtube.SCOPES)
flow.redirect_uri = "http://localhost:8765/"
url, _ = flow.authorization_url(prompt="consent", access_type="offline")
print("AUTH_URL", url, flush=True)
webbrowser.open(url)
creds = flow.run_local_server(port=8765, open_browser=False, prompt="consent", access_type="offline",
                              success_message="Authorised. You can close this tab.")
Path("C:/disaster-maps/youtube_token.json").write_text(json.dumps({"refresh_token": creds.refresh_token, "client_id": creds.client_id,
    "client_secret": creds.client_secret, "token_uri": creds.token_uri, "scopes": youtube.SCOPES}), encoding="utf-8")
print("TOKEN_SAVED", flush=True)
