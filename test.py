layers_hidden = [2, 2, 1]

for in_features, out_features in zip(layers_hidden, layers_hidden[1:]):
    print(in_features, out_features)
