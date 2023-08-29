from pyngrok import ngrok,conf

port=5000
conf.get_default().auth_token=TOKEN # Supply token here using loaddotenv
public_url = ngrok.connect(port).public_url
print(" * ngrok tunnel \"{}\" -> \"http://127.0.0.1:{}\"".format(public_url, port))
