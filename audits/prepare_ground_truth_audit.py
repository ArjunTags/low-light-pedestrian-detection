from pathlib import Path
import csv

PROJECT_ROOT = Path('/home/arjun/lli-pedestrian')
IMAGE_DIR = PROJECT_ROOT / 'data' / 'yolo' / 'images' / 'test'
LABEL_DIR = PROJECT_ROOT / 'data' / 'yolo' / 'labels' / 'test'
OUTPUT = Path(__file__).resolve().parent / 'test_ground_truth_audit.csv'
IMAGE_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}


def count_label_objects(label_path: Path) -> int:
    if not label_path.exists():
        return 0
    return sum(
        1
        for line in label_path.read_text(encoding='utf-8').splitlines()
        if line.strip() and not line.lstrip().startswith('#')
    )


def main() -> None:
    if not IMAGE_DIR.exists():
        raise FileNotFoundError(f'Test image directory not found: {IMAGE_DIR}')
    if not LABEL_DIR.exists():
        raise FileNotFoundError(f'Test label directory not found: {LABEL_DIR}')

    images = sorted(
        path for path in IMAGE_DIR.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )
    if not images:
        raise FileNotFoundError(f'No test images found in: {IMAGE_DIR}')

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT.open('w', newline='', encoding='utf-8') as handle:
        writer = csv.writer(handle)
        writer.writerow([
            'image_name',
            'ground_truth_count',
            'annotated_count',
            'verified',
            'notes',
        ])
        for image_path in images:
            label_path = LABEL_DIR / f'{image_path.stem}.txt'
            writer.writerow([
                image_path.name,
                '',
                count_label_objects(label_path),
                '',
                '',
            ])

    missing_labels = sum(
        1 for image_path in images
        if not (LABEL_DIR / f'{image_path.stem}.txt').exists()
    )
    print(f'Created: {OUTPUT}')
    print(f'Test images prepared: {len(images)}')
    print(f'Missing label files: {missing_labels}')
    print('Fill in ground_truth_count, verified, and notes after visual inspection.')


if __name__ == '__main__':
    main()
