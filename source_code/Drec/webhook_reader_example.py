from flask import Flask, request


stateStore = []

app = Flask(__name__)


@app.route('/webhookcallback/sleepstate', methods=['POST'])
def sleepStateHook():
    global stateStore
    state = request.values.get('state')
    epoch = request.values.get('epoch')

    stateStore.append((epoch, state))

    print(f'state: {state}')
    print('epoch: ' + str(epoch))

    return "received"


@app.route('/webhookcallback/hello', methods=['POST'])
def helloHook():
    msg = request.values.get('hello')
    print(f'hello message sent. message: {msg}')

    return "received"


@app.route('/webhookcallback/finished', methods=['POST'])
def recordingFinishedHook():
    global stateStore
    lines = [', '.join([str(epoch), str(state)]) for epoch, state in stateStore]
    with open('received_sleep_states.txt', 'w') as f:
        f.writelines(lines)
    stateStore = []

    return "received"


if __name__ == '__main__':
    app.run()
