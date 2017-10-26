# Content

Toy project which intends to tag frames from Youtube videos with a label explaining its content.

What it does:

1. Download videos from Youtube
2. Extract frames from the downloaded videos
3. Tag specific frames with its content

The proposed algorithm assign tags to specific frames even when the input dataframe only contains a set of tags for the whole video (no frame per frame annotated video).

# Status

The proposed code works properly in sequences generated on top of the MNIST dataset.
It does not work as expected in Youtube videos. The computing power I have access to does not allow me to properly test the proposed network and iterate.

