if [ -z "${GDRIVE_CLIENT_ID+x}" ]; then
  echo 'GDRIVE_CLIENT_ID env variable is not set 😭😭😭! Pwease, initialize it'
  exit 1
fi
if [ -z "${GDRIVE_CLIENT_SECRET+x}" ]; then
  echo 'GDRIVE_CLIENT_SECRET env variable is not set 💀💀💀! Please, initialize it'
  exit 1
fi
dvc remote modify --local gdrive_remote gdrive_client_id $GDRIVE_CLIENT_ID
dvc remote modify --local gdrive_remote gdrive_client_secret $GDRIVE_CLIENT_SECRET
echo 'gdrive_client_id and gdrive_client_secret were successfully initialized 😉'
