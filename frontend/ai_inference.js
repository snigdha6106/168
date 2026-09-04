// Pure JavaScript LSTM and Linear Layer Implementation
import modelParams from './assets/model_parameters.json';

const sigmoid = (x) => 1 / (1 + Math.exp(-x));
const tanh = (x) => Math.tanh(x);

// Matrix-Vector multiplication
function matVecMul(matrix, vector) {
    const result = new Array(matrix.length).fill(0);
    for (let i = 0; i < matrix.length; i++) {
        for (let j = 0; j < vector.length; j++) {
            result[i] += matrix[i][j] * vector[j];
        }
    }
    return result;
}

// Vector addition
function vecAdd(v1, v2) {
    return v1.map((val, i) => val + v2[i]);
}

// Element-wise multiplication
function vecMul(v1, v2) {
    return v1.map((val, i) => val * v2[i]);
}

// Run one step of LSTM
function lstmStep(x, h_prev, c_prev, w_ih, w_hh, b_ih, b_hh, hiddenSize) {
    const gates_ih = matVecMul(w_ih, x);
    const gates_hh = matVecMul(w_hh, h_prev);
    
    // Add biases and sum ih + hh
    const gates = new Array(4 * hiddenSize);
    for (let i = 0; i < 4 * hiddenSize; i++) {
        gates[i] = gates_ih[i] + b_ih[i] + gates_hh[i] + b_hh[i];
    }
    
    const i_t = new Array(hiddenSize);
    const f_t = new Array(hiddenSize);
    const g_t = new Array(hiddenSize);
    const o_t = new Array(hiddenSize);
    
    for (let i = 0; i < hiddenSize; i++) {
        i_t[i] = sigmoid(gates[i]);
        f_t[i] = sigmoid(gates[hiddenSize + i]);
        g_t[i] = tanh(gates[2 * hiddenSize + i]);
        o_t[i] = sigmoid(gates[3 * hiddenSize + i]);
    }
    
    const c_t = new Array(hiddenSize);
    const h_t = new Array(hiddenSize);
    for (let i = 0; i < hiddenSize; i++) {
        c_t[i] = f_t[i] * c_prev[i] + i_t[i] * g_t[i];
        h_t[i] = o_t[i] * Math.tanh(c_t[i]);
    }
    
    return { h: h_t, c: c_t };
}

export class AILocalEngine {
    constructor() {
        this.hiddenSize = 32;
        this.seqLen = 10;
        this.buffer = [];
        this.params = modelParams;
    }

    // Process a single IMU frame [acc_x, acc_y, acc_z, gyro_x, gyro_y, gyro_z]
    predict(imuData) {
        // Normalize
        const normData = imuData.map((val, i) => 
            (val - this.params.X_mean[i]) / (this.params.X_std[i] + 1e-8)
        );
        
        this.buffer.push(normData);
        if (this.buffer.length > this.seqLen) {
            this.buffer.shift();
        }
        
        if (this.buffer.length < this.seqLen) {
            return this.params.y_mean;
        }

        // Run LSTM over the sequence
        let h0 = new Array(this.hiddenSize).fill(0);
        let c0 = new Array(this.hiddenSize).fill(0);
        
        let h1 = new Array(this.hiddenSize).fill(0);
        let c1 = new Array(this.hiddenSize).fill(0);
        
        for (let t = 0; t < this.seqLen; t++) {
            const x_t = this.buffer[t];
            
            // Layer 0
            const state0 = lstmStep(
                x_t, h0, c0,
                this.params.lstm_weight_ih_l0, this.params.lstm_weight_hh_l0,
                this.params.lstm_bias_ih_l0, this.params.lstm_bias_hh_l0,
                this.hiddenSize
            );
            h0 = state0.h;
            c0 = state0.c;
            
            // Layer 1
            const state1 = lstmStep(
                h0, h1, c1,
                this.params.lstm_weight_ih_l1, this.params.lstm_weight_hh_l1,
                this.params.lstm_bias_ih_l1, this.params.lstm_bias_hh_l1,
                this.hiddenSize
            );
            h1 = state1.h;
            c1 = state1.c;
        }
        
        // Final Linear Layer
        let out = 0;
        for (let i = 0; i < this.hiddenSize; i++) {
            out += h1[i] * this.params.fc_weight[0][i];
        }
        out += this.params.fc_bias[0];
        
        // Denormalize
        return (out * this.params.y_std) + this.params.y_mean;
    }
}
