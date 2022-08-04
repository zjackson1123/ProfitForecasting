import os 
from pages import app


if __name__ == "__main__":
    HOST = os.environ.get('SERVER_HOST', 'localhost')

    try:
        PORT = int(os.environ.get('SERVER_PORT', '4449'))
    
    except:
        PORT = 4449

    app.run(HOST, PORT)

