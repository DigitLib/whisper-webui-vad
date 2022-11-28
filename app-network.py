# Run the app with no audio file restrictions, and make it available on the network
from app import create_ui
create_ui(-1, server_name="0.0.0.0")