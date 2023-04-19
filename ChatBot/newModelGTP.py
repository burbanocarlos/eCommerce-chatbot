import json
import random

from rasa.core.agent import Agent
from rasa.shared.nlu.training_data import Message
from rasa.core.utils import EndpointConfig
from rasa.utils.endpoints import EndpointConfig
from rasa.core.channels.socketio import SocketIOInput
from sanic import Sanic, response
from sanic.request import Request
from sanic.response import HTTPResponse
from typing import Text, Dict, Any, List

# define constants
ENDPOINT = "http://localhost:5055/webhook"
MODEL_PATH = "models/nlu"

# initialize interpreter and agent
interpreter = Message(MODEL_PATH)
agent = Agent.load("models/dialogue", interpreter=interpreter)

# define socketio input channel
input_channel = SocketIOInput(
    # event name for messages sent from the user
    user_message_evt="user_uttered",
    # event name for messages sent from the bot
    bot_message_evt="bot_uttered",
    # socket.io namespace to use for the channel
    namespace=None
)

# create Sanic app
app = Sanic()

# define handler for incoming messages
@app.route('/webhook', methods=['POST'])
async def webhook(request: Request) -> HTTPResponse:
    # parse incoming message
    message = request.json.get('message')

    # get response from bot
    response = await agent.handle_text(message, output_channel=input_channel)

    # return response to client
    return response[0].get('text')

if __name__ == '__main__':
    app.run(host='localhost', port=5000, debug=True)
