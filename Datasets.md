# Datasets

## Types

- Brain feature: Tensor (1, 420)
- Visual features compressed: List (768)
- Visual features uncompressed:
- labels: 40

## Signals to brain Representations

### Columns

```python
data = {"features": [], "images": [], "labels": []}
```

3 lists of equal length columns.

Features: The brain features extracted from each signal using our modified AttnSleep model
images: image names list
labels: list of labels

## Images to visual representation

```python
data = {
  "image_name.JPG": <visual-features>,
  ... 
}
```

A dictionary that maps each image to its visual features extracted by a ViT model.
This can be either compressed or uncompressed.
Compressed visual features have an average applied to each batch data so it's smaller in size.

## Visual to brain representations

filename: visual-to-brain-features.pth

```python
data = [
  {
    image_name: "image_name.JPG",
    label: "Dog", # will probably be the number rather than the name
    visual_features: np.array([...]),
    brain_features: np.array([...]),
  }
]
```
