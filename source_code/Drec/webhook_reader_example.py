from flask import Flask, request


stateStore = []

app = Flask(__name__)


@app.route('/webhookcallback/sleepstate', methods=['POST'])
def sleepStateHook():
    global stateStore
    time_dreamento = request.values.get('time_dreamento')
    scoring_dreamento = request.values.get('scoring_dreamento')
    epoch_dreamento = request.values.get('epoch_dreamento')
    epoch_yasa = request.values.get('epoch_yasa')
    time_yasa = request.values.get('time_yasa')
    rem_by_yasa_scoring = request.values.get('rem_by_yasa_scoring')
    rem_by_yasa_eyes = request.values.get('rem_by_yasa_eyes')

    stateStore.append((time_dreamento, scoring_dreamento, epoch_dreamento))


    print(f'dreamento: {epoch_dreamento} - {time_dreamento} - {str(scoring_dreamento)}')
    print(f'yasa: {epoch_yasa} - {time_yasa} - {str(rem_by_yasa_scoring)} - {rem_by_yasa_eyes}')

    return "received"


@app.route('/webhookcallback/hello', methods=['POST'])
def helloHook():
    msg = request.values.get('hello')
    print(f'hello message sent. message: {msg}')

    return "received"


@app.route('/webhookcallback/finished', methods=['POST'])
def recordingFinishedHook():
    global stateStore
    lines = [', '.join([str(time), str(epoch), str(scoring_dreamento),]) for time, scoring_dreamento, epoch in stateStore]
    with open('received_sleep_states.txt', 'w') as f:
        f.writelines(lines)
    stateStore = []

    return "received"


if __name__ == '__main__':
    app.run(host="127.0.0.1", port=5000)
    # app.run(host="127.0.0.1", port=3000)
