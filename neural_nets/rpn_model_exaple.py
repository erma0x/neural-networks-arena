feature_map_tile = Input(shape=(None,None,1536))
convolution_3x3 = Conv2D(
    filters=512,
    kernel_size=(3, 3),
    name="3x3"
)(feature_map_tile)

output_deltas = Conv2D(
    filters= 4 * k,
    kernel_size=(1, 1),
    activation="linear",
    kernel_initializer="uniform",
    name="deltas1"
)(convolution_3x3)

output_scores = Conv2D(
    filters=1 * k,
    kernel_size=(1, 1),
    activation="sigmoid",
    kernel_initializer="uniform",
    name="scores1"
)(convolution_3x3)

model = Model(inputs=[feature_map_tile], outputs=[output_scores, output_deltas])
