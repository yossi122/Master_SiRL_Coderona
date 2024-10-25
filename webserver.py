import os
from flask import Flask
from flask_autoindex import AutoIndex

app = Flask(__name__)
AutoIndex(app, browse_root="./")
print('current directory is:  ' + os.getcwd() + ' !!!!!!!!!!!')
app.run(host='0.0.0.0', port=8080, debug=False)

@app.route('/hello')
def hello():
    """Return a friendly HTTP greeting."""
    return 'coderona on Google App Engine. Use /static to see the output files'

