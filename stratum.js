const net = require('net');
const EventEmitter = require('events');

class StratumClient extends EventEmitter {
    constructor(host, port, username, password) {
        super();
        this.host = host;
        this.port = port;
        this.username = username;
        this.password = password;
        this.client = new net.Socket();
        this.nextId = 1;
    }

    connect() {
        this.client.connect(this.port, this.host, () => {
            console.log('Connected to Stratum server');
            this.subscribe();
        });

        this.client.on('data', (data) => {
            this.handleResponse(data);
        });

        this.client.on('close', () => {
            console.log('Connection closed');
        });

        this.client.on('error', (err) => {
            console.error('Connection error:', err);
        });
    }

    send(method, params) {
        const message = JSON.stringify({
            id: this.nextId++,
            method: method,
            params: params
        }) + '\n';
        this.client.write(message);
    }

    subscribe() {
        this.send('mining.subscribe', []);
    }

    authorize() {
        this.send('mining.authorize', [this.username, this.password]);
    }

    submit(jobId, nonce, result) {
        this.send('mining.submit', [this.username, jobId, nonce, result]);
    }

    handleResponse(data) {
        const responses = data.toString().split('\n').filter(Boolean);
        responses.forEach(response => {
            const message = JSON.parse(response);
            if (message.id === 1) {
                this.authorize();
            } else if (message.method === 'mining.notify') {
                this.emit('job', message.params);
            } else if (message.result || message.error) {
                console.log('Server response:', message);
            }
        });
    }
}

module.exports = StratumClient;