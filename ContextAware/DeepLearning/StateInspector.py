# Assuming your model class is defined as BidirectionalLSTMLaneReconstructor

class StateInspector(nn.Module):
    def __init__(self, original_model):
        super(StateInspector, self).__init__()
        self.model = original_model

    def forward(self, x):
        batch_size, num_lane_unit, time_span = x.size()
        x = x.view(batch_size, time_span, num_lane_unit)

        # Encode the input sequence
        encoder_outputs, (hidden, cell) = self.model.encoder(x)

        # Prepare hidden and cell states for the decoder
        hidden = hidden.view(self.model.num_layers, 2, batch_size, self.model.hidden_size)
        hidden = torch.cat([hidden[:, 0, :, :], hidden[:, 1, :, :]], dim=2)
        cell = cell.view(self.model.num_layers, 2, batch_size, self.model.hidden_size)
        cell = torch.cat([cell[:, 0, :, :], cell[:, 1, :, :]], dim=2)

        # Store states for inspection
        all_hidden_states = [hidden]
        all_cell_states = [cell]

        # Decode step by step
        decoder_input = encoder_outputs
        for t in range(time_span):
            decoder_output, (hidden, cell) = self.model.decoder(
                decoder_input[:, t:t+1, :], (hidden, cell)
            )
            all_hidden_states.append(hidden)
            all_cell_states.append(cell)

        # Apply output layer and sigmoid
        outputs = self.model.output_layer(decoder_output)
        reconstructed = torch.sigmoid(outputs)
        reconstructed = reconstructed.reshape(-1, num_lane_unit, time_span)

        return reconstructed, all_hidden_states, all_cell_states

# Function to run inference and inspect states
def inspect_states(model, sample_input):
    inspector = StateInspector(model)
    inspector.eval()
    with torch.no_grad():
        reconstructed, hidden_states, cell_states = inspector(sample_input)
    
    return reconstructed, hidden_states, cell_states

# Function to visualize states
def visualize_states(hidden_states, cell_states):
    num_timesteps = len(hidden_states)
    hidden_norms = [torch.norm(h, dim=2).mean().item() for h in hidden_states]
    cell_norms = [torch.norm(c, dim=2).mean().item() for c in cell_states]

    plt.figure(figsize=(12, 6))
    plt.plot(range(num_timesteps), hidden_norms, label='Hidden State Norm')
    plt.plot(range(num_timesteps), cell_norms, label='Cell State Norm')
    plt.xlabel('Time Step')
    plt.ylabel('Average Norm')
    plt.title('LSTM State Norms Over Time')
    plt.legend()
    plt.show()
