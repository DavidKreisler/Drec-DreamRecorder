from flask import Flask, request


stateStore = []

app = Flask(__name__)


@app.route('/webhookcallback/sleepstate', methods=['POST'])
def sleepStateHook():
    global stateStore
    time = request.values.get('time')
    epoch = request.values.get('epoch')
    rem_by_scoring = request.values.get('rem_by_scoring')
    rem_by_eyes = request.values.get('rem_by_eyes')
    scoring_dreamento = request.values.get('scoring_dreamento')
    epoch_dreamento = request.values.get('epoch_dreamento')

    stateStore.append((time, epoch, rem_by_scoring, rem_by_eyes, scoring_dreamento, epoch_dreamento))

    print(f'rem_by_scoring: {rem_by_scoring}')
    print(f'rem_by_eyes: {rem_by_eyes}')
    print(f'scoring_dreamento: {scoring_dreamento}')
    print('epoch / dreamento epoch: ' + str(epoch), str(epoch_dreamento))

    return "received"


@app.route('/webhookcallback/hello', methods=['POST'])
def helloHook():
    msg = request.values.get('hello')
    print(f'hello message sent. message: {msg}')

    return "received"


@app.route('/webhookcallback/finished', methods=['POST'])
def recordingFinishedHook():
    global stateStore
    lines = [', '.join([str(time), str(epoch), str(rem_by_staging_and_eyes), str(rem_by_powerbands)]) for time, epoch, rem_by_staging_and_eyes, rem_by_powerbands in stateStore]
    with open('received_sleep_states.txt', 'w') as f:
        f.writelines(lines)
    stateStore = []

    return "received"


if __name__ == '__main__':
    app.run(host="127.0.0.1", port=5000)
