from flask import Flask, request


stateStore = []

app = Flask(__name__)


@app.route('/webhookcallback/sleepstate', methods=['POST'])
def sleepStateHook():
    global stateStore
    rem_by_staging_and_eyes = request.values.get('rem_by_staging_and_eyes')
    rem_by_powerbands = request.values.get('rem_by_powerbands')
    time = request.values.get('time')
    epoch = request.values.get('epoch')

    stateStore.append((time, epoch, rem_by_staging_and_eyes, rem_by_powerbands))

    print(f'rem_by_staging_and_eyes: {rem_by_staging_and_eyes}')
    print(f'rem_by_powerbands: {rem_by_powerbands}')
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
    lines = [', '.join([str(time), str(epoch), str(rem_by_staging_and_eyes), str(rem_by_powerbands)]) for time, epoch, rem_by_staging_and_eyes, rem_by_powerbands in stateStore]
    with open('received_sleep_states.txt', 'w') as f:
        f.writelines(lines)
    stateStore = []

    return "received"


if __name__ == '__main__':
    app.run()
