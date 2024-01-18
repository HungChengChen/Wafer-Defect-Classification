import torchvision.transforms as T


def BASIC_TRANSFORMS(
    img_size: list = [224, 224],
    mean: list = [0.485, 0.456, 0.406],
    std: list = [0.229, 0.224, 0.225],
):
    return T.Compose(
        [
            T.Resize(img_size),
            T.ToTensor(),
            T.Normalize(
                mean=mean,  # RGB
                std=std,
            ),
        ]
    )


def AUGMENTATION_TRANSFORMS(
    img_size: list = [224, 224],
    mean: list = [0.485, 0.456, 0.406],
    std: list = [0.229, 0.224, 0.225],
):
    return T.Compose(
        [
            T.Resize(img_size),
            T.RandomRotation((-45, 45), fill=(255)),
            T.RandomHorizontalFlip(p=0.5),
            T.RandomVerticalFlip(p=0.5),
            T.ToTensor(),
            T.Normalize(mean=mean, std=std),
        ]
    )
