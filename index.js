const Miner = require('./miner');

const host = 'stratum+tcp://losmuchachos.digital'; // Replace with actual Stratum server host
const port = 8023; // Replace with actual Stratum server port
const username = 'xel:qzr5sxxnrk4mh5p5j43s854rnypvu2nu8vg5yup0ydqzuerrpdrsqpfe0hk.XPTO'; // Replace with your Stratum username
const password = 'x'; // Replace with your Stratum password

const miner = new Miner(host, port, username, password);

console.log(miner);