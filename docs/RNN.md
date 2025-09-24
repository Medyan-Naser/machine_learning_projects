### 2. Recurrent neural networks (RNN)
Recurrent neural networks (RNNs) are specially designed to handle sequential data, making them a powerful tool for applications like language modeling and time series forecasting. The essence of their design lies in maintaining a 'memory' or 'hidden state' throughout the sequence by employing loops. This enables RNNs to recognize and capture the temporal dependencies inherent in sequential data.
- Hidden state: Often referred to as the network's 'memory', the hidden state is a dynamic storage of information about previous sequence inputs. With each new input, this hidden state is updated, factoring in both the new input and its previous value.
- Temporal dependency: Loops in RNNs enable information transfer across sequence steps.

### 3. Long short-term memory (LSTM) and gated recurrent units (GRUs)
Long short-term memory (LSTM) and gated recurrent units (GRUs) are advanced variations of recurrent neural networks (RNNs), designed to address the limitations of traditional RNNs and enhance their ability to model sequential data effectively. They processed sequences one element at a time and maintained an internal state to remember past elements. While they were effective for a variety of tasks, they struggled with long sequences and long-term dependencies.


In practice, modifications and enhancements, such as Long Short-Term Memory (LSTM) and Gated Recurrent Units (GRU), are often used to address issues like the vanishing gradient problem in basic RNNs.

An LSTM cell has three main components: an input gate, a forget gate, and an output gate.
- The **input gate** controls how much new information should be stored in the cell's memory. It looks at the current input and the previous hidden state and decides which parts of the new input to remember.
- The **forget gate** determines what information should be discarded or forgotten from the cell's memory. It considers the current input and the previous hidden state and decides which parts of the previous memory are no longer relevant.
- The **output gate** determines what information should be outputted from the cell. It looks at the current input and the previous hidden state and decides which parts of the cell's memory to include in the output.

The key idea behind LSTM cells is that they have a separate memory state that can selectively retain or forget information over time. This helps them handle long-range dependencies and remember important information from earlier steps in a sequence.