# Backpropagation

$LEARNING_RATE = 0.7
$MOMENTUM = 0.3

# Sigmoid function
def activation (input)
  1 / (1 + Math.exp(-input))
end

# Derivative of sigmoid function
def d_activation (input)
  s = activation(input)
  s * (1.0 - s)
end

class Neuron
  attr_accessor :input_connections, :output_connections, :delta, :output
  attr_reader :layer, :sum, :norm

  def initialize (layer)
    @input_connections = Array.new
    @output_connections = Array.new
    @delta = 0
    @output = 1.0
    @layer = layer
    @sum = 0
  end

  def calculate_delta
    if !@layer.is_output_layer and !@layer.is_input_layer # Skip output (already done) and input (no need). This also skips biases
      @delta = 0
      for connection in @output_connections
        neuron = connection.output_neuron
        @delta += neuron.delta * connection.weight
      end

      @delta *= d_activation(@sum)
    end
  end

  def calculate_norm
    @norm = 0
    for connection in @input_connections
      @norm  += connection.weight ** 2
    end

    @norm = Math.sqrt(@norm)
  end

  def pulse
    @output = 0
    @sum = 0
    for connection in @input_connections
      @sum += connection.calculate_value
    end

    # Get bias connection
    for connection in @layer.bias.output_connections
      if connection.output_neuron == self
        @sum += connection.calculate_value
        break
      end
    end

    @output = activation(@sum)
  end

  def generate_output_connections (next_layer)
    for neuron in next_layer.neurons
      connection = Connection.new(self, neuron)

      neuron.input_connections << connection
      @output_connections << connection
    end
  end
end

class Connection
  attr_reader :weight, :gradient, :delta_change, :input_neuron, :output_neuron

  def initialize (input_neuron, output_neuron)
    @input_neuron = input_neuron
    @output_neuron = output_neuron
    @weight = rand(-1.0..1.0)
    @delta_change = 0
    @gradient = 0
  end

  def calculate_value
    @input_neuron.output * @weight
  end

  def initial_weight_adjust (beta)
    @weight = (beta * @weight) / @output_neuron.norm
  end

  def calculate_gradient
    @gradient += @input_neuron.output * @output_neuron.delta
  end

  def calculate_delta_change
    @delta_change = ($LEARNING_RATE * @gradient) + ($MOMENTUM * @delta_change)
    @gradient = 0
  end

  def update_weight
    @weight += @delta_change
  end
end

class Layer
  attr_reader :neurons, :bias, :is_input_layer, :is_output_layer

  def initialize (neuron_count, is_input_layer, is_output_layer)
    @is_input_layer = is_input_layer
    @is_output_layer = is_output_layer

    generate_neurons(neuron_count)

    if !@is_input_layer
      generate_bias
    end
  end

  def pulse
    if !@is_input_layer
      for neuron in @neurons
        neuron.pulse
      end
    end
  end

  def generate_connections (next_layer)
    if !is_output_layer # Don't create output connections if there is no neuron to connect to
      for neuron in @neurons
        neuron.generate_output_connections(next_layer)
      end
    end
  end

  private
  def generate_neurons (neuron_count)
    @neurons = Array.new

    i = 0
    while i < neuron_count
      @neurons[i] = Neuron.new(self)
      i += 1
    end
  end

  def generate_bias
    @bias = Neuron.new(self)
    @bias.output = 1.0
    @bias.generate_output_connections(self)
  end
end

class Network
  attr_reader :layers, :global_error, :neurons, :connections, :update_after

  def initialize (layer_count, input_neuron_count, hidden_neuron_count, output_neuron_count, update_after)
    @global_error = 1
    @update_after = update_after
    @on_update = 0
    generate_layers(layer_count, input_neuron_count, hidden_neuron_count, output_neuron_count)
    generate_connections

    get_all_neurons
    get_all_connections

    calculate_beta(input_neuron_count, hidden_neuron_count)
    initial_weight_adjust
  end

  def pulse
    for layer in @layers
      layer.pulse
    end
  end

  def train (inputs, desired_results)
    @layers.first.neurons.each_with_index do |neuron, index|
      neuron.output = inputs[index]
    end

    pulse
    calculate_global_error(desired_results)

    @on_update += 1
    back_propogation(desired_results)
  end

  private
  def back_propogation (desired_results)
    calculate_deltas(desired_results)

    for connection in @connections
      connection.calculate_gradient

      if @on_update >= @update_after
        connection.calculate_delta_change
        connection.update_weight
        on_update = 0
      end
    end
  end

  def calculate_beta (input_neuron_count, hidden_neuron_count)
    @beta = 0.7 * (hidden_neuron_count ** (1.0 / input_neuron_count))
  end

  def calculate_deltas (desired_results)
    # Calculate deltas for output layer
    layers.last.neurons.each_with_index do |neuron, index|
      error = neuron.output - desired_results[index]
      neuron.delta = -error * d_activation(neuron.sum)
    end
    
    # Calculate deltas for other layers
    for neuron in @neurons
      neuron.calculate_delta
    end
  end

  def calculate_global_error (desired_results)
    sum = 0
    @layers.last.neurons.each_with_index do |neuron, index|
      sum += (desired_results[index] - neuron.output) ** 2.0
    end

    @global_error = sum / desired_results.length
  end

  def get_all_neurons
    @neurons = Array.new

    for layer in @layers
      for neuron in layer.neurons
        @neurons << neuron
      end
    end
  end

  def get_all_connections
    @connections = Array.new

    for layer in @layers
      for neuron in layer.neurons
        if !layer.is_output_layer
          for connection in neuron.output_connections
            @connections << connection
          end
        end
      end

      bias = layer.bias
      if !layer.is_input_layer
        for connection in bias.output_connections
          @connections << connection
        end
      end
    end
  end

  def generate_layers (layer_count, input_neuron_count, hidden_neuron_count, output_neuron_count)
    @layers = Array.new
    @layers[0] = Layer.new(input_neuron_count, true, false)
    @layers[layer_count - 1] = Layer.new(output_neuron_count, false, true)

    # Skip input and output layers
    i = 1
    while i < layer_count - 1
      @layers[i] = Layer.new(hidden_neuron_count, false, false)
      i += 1
    end
  end

  def generate_connections
    @layers.each_with_index do |layer, index|
      layer.generate_connections(layers[index + 1])
    end
  end

  def initial_weight_adjust
    for neuron in @layers[1].neurons # Only do first hidden layer
      neuron.calculate_norm

      for connection in neuron.input_connections
        connection.initial_weight_adjust(@beta)
      end
    end

    @layers[1].bias.calculate_norm # Bias neurons for n layer are a part of n layer
    for connection in @layers[1].bias.output_connections
      connection.initial_weight_adjust(@beta)
    end
  end
end