# Synthetic media fixtures

`red-square.png` is a red square on a white background. `color-cuts.mp4` uses the same square and changes its fill from red to blue at 4 seconds, then to green at 8 seconds. The video is 12 seconds long at 12 fps, 384 × 216, H.264/yuv420p, with no audio. Both files were generated with FFmpeg color and drawbox filters. The PNG is the first decoded frame of the video.

```text
40381afa73784fbb0b2858b94dbc17dca3e54dec629d08958787e16e446f3498  red-square.png
f7b3fc8e088fe908098d03daa54858058ada76bae941a0e24b10e582be148db9  color-cuts.mp4
```

The files are covered by the [repository’s MIT license](../LICENSE).
