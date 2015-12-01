require_relative 'neural_network.rb'

def generate_xor
  inputs = [rand(2), rand(2)]
  correct = [inputs[0] ^ inputs[1]]

  return inputs, correct
end

def test (network)
  i = 0

  while i < 20
    inputs, correct = generate_xor

    network.layers.first.neurons[0].output = inputs[0]
    network.layers.first.neurons[1].output = inputs[1]
    network.pulse

    output = network.layers.last.neurons[0].output.round

    puts "Input: " + inputs.to_s
    puts "Correct Output: " + correct[0].to_s
    puts "Network Output: " + output.to_s
    puts "Error: " + (network.global_error * 100.0).to_s + "%"

    if output != correct[0]
      return false
    end

    i += 1
  end

  return true
end

network = Network.new(3, 2, 3, 1)

times_trained = 0
while true
  inputs, correct = generate_xor
  network.train(inputs, correct)
  output = network.layers.last.neurons[0].output.round

  puts "Input: " + inputs.to_s
  puts "Correct Output: " + correct[0].to_s
  puts "Network Output: " + output.to_s
  puts "Error: " + (network.global_error * 100.0).to_s + "%"

  times_trained += 1

  if network.global_error < 0.1
    if test(network)
      break
    end
  end
end

puts "Training complete"
puts "It took " + times_trained.to_s + " pulses"