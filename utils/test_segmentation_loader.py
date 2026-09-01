from segmentation_loader import SegmentationLoader


loader = SegmentationLoader()

images, masks = loader.load_data(
    max_samples=5
)

print()
print("=" * 50)
print("TEST SUCCESSFUL")
print("=" * 50)

print("Images shape:", images.shape)
print("Masks shape :", masks.shape)

print("First image shape:", images[0].shape)
print("First mask shape :", masks[0].shape)

print("=" * 50)