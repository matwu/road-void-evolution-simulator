"""
Convert gprMax output files to CSV
"""
import sys
import argparse
import csv
from pathlib import Path
import h5py
import numpy as np


def load_gpr_output(output_file: str) -> dict:
    """
    Load gprMax output file

    Args:
        output_file: .out file path

    Returns:
        Data dictionary
    """
    data = {}

    with h5py.File(output_file, 'r') as f:
        # Get receiver data
        rxs = f['rxs']

        # Load data for each receiver
        for rx_name in rxs.keys():
            rx = rxs[rx_name]
            rx_data = {}

            # Load each component
            for component in ['Ex', 'Ey', 'Ez', 'Hx', 'Hy', 'Hz']:
                if component in rx:
                    rx_data[component] = np.array(rx[component])

            data[rx_name] = rx_data

        # Metadata
        if 'dt' in f.attrs:
            data['dt'] = f.attrs['dt']
        if 'iterations' in f.attrs:
            data['iterations'] = f.attrs['iterations']

    return data


def export_to_csv(output_file: str, csv_file: str, rx_name: str = 'rx1'):
    """
    Convert gprMax output to CSV file

    Args:
        output_file: .out file path
        csv_file: Output CSV file path
        rx_name: Receiver name
    """
    data = load_gpr_output(output_file)

    if rx_name not in data:
        print(f"Error: Receiver {rx_name} not found")
        available = [k for k in data.keys() if k.startswith('rx')]
        if available:
            print(f"Available receivers: {available}")
        return False

    rx_data = data[rx_name]

    # Create time axis
    n_samples = len(rx_data[list(rx_data.keys())[0]])

    if 'dt' in data:
        dt = data['dt']
        time = np.arange(n_samples) * dt
        time_ns = time * 1e9  # ns
    else:
        time = np.arange(n_samples)
        time_ns = time

    # Prepare CSV data
    components = ['Ex', 'Ey', 'Ez', 'Hx', 'Hy', 'Hz']
    available_components = [c for c in components if c in rx_data]

    # CSV output
    csv_path = Path(csv_file)
    csv_path.parent.mkdir(parents=True, exist_ok=True)

    with open(csv_path, 'w', newline='') as f:
        writer = csv.writer(f)

        # Header
        header = ['time_ns'] + available_components
        writer.writerow(header)

        # Data rows
        for i in range(n_samples):
            row = [time_ns[i]]
            for component in available_components:
                row.append(rx_data[component][i])
            writer.writerow(row)

    print(f"Exported to {csv_path}")
    print(f"  Rows: {n_samples}")
    print(f"  Columns: {header}")

    return True


def export_batch(input_dir: str, output_dir: str, pattern: str = '*.out'):
    """
    Batch convert all .out files in directory to CSV

    Args:
        input_dir: Input directory
        output_dir: Output directory
        pattern: File pattern
    """
    input_path = Path(input_dir)
    output_path = Path(output_dir)

    if not input_path.exists():
        print(f"Error: {input_path} does not exist")
        return

    # Search for .out files
    out_files = sorted(input_path.glob(pattern))

    if not out_files:
        print(f"No .out files found in {input_path}")
        return

    print(f"Found {len(out_files)} .out files")
    print(f"Exporting to {output_path}...")

    success_count = 0
    for out_file in out_files:
        csv_file = output_path / f"{out_file.stem}.csv"

        try:
            if export_to_csv(str(out_file), str(csv_file)):
                success_count += 1
        except Exception as e:
            print(f"Error processing {out_file}: {e}")

    print(f"\nExport completed: {success_count}/{len(out_files)} files")


def export_sequence_to_single_csv(
    output_files: list,
    csv_file: str,
    component: str = 'Ez',
    rx_name: str = 'rx1'
):
    """
    Combine multiple .out files into a single CSV file (for time-series data)

    Args:
        output_files: List of .out files
        csv_file: Output CSV file path
        component: Electric/magnetic field component
        rx_name: Receiver name
    """
    # CSV output
    csv_path = Path(csv_file)
    csv_path.parent.mkdir(parents=True, exist_ok=True)

    total_rows = 0
    file_count = 0

    with open(csv_path, 'w', newline='') as f:
        writer = csv.writer(f)

        # Header
        writer.writerow(['sequence_id', 'file', 'time_ns', component])

        # Write data for each file
        for i, out_file in enumerate(sorted(output_files)):
            data = load_gpr_output(out_file)

            if rx_name not in data or component not in data[rx_name]:
                print(f"Warning: {component} not found in {out_file}")
                continue

            signal = data[rx_name][component]

            # Time axis
            n_samples = len(signal)
            if 'dt' in data:
                dt = data['dt']
                time_ns = np.arange(n_samples) * dt * 1e9
            else:
                time_ns = np.arange(n_samples)

            # Write data rows
            filename = Path(out_file).name
            for j in range(n_samples):
                writer.writerow([i, filename, time_ns[j], signal[j]])
                total_rows += 1

            file_count += 1

    if file_count == 0:
        print("Error: No data to export")
        return False

    print(f"Exported combined data to {csv_path}")
    print(f"  Rows: {total_rows}")
    print(f"  Files: {file_count}")

    return True


def main():
    """Main execution"""
    parser = argparse.ArgumentParser(description='Export gprMax output to CSV')
    parser.add_argument(
        'input',
        type=str,
        help='Input .out file or directory'
    )
    parser.add_argument(
        '--output',
        type=str,
        required=True,
        help='Output CSV file or directory'
    )
    parser.add_argument(
        '--mode',
        type=str,
        default='single',
        choices=['single', 'batch', 'sequence'],
        help='Export mode: single (one file), batch (directory), sequence (combine multiple)'
    )
    parser.add_argument(
        '--component',
        type=str,
        default='Ez',
        help='Field component for sequence mode (default: Ez)'
    )
    parser.add_argument(
        '--rx',
        type=str,
        default='rx1',
        help='Receiver name (default: rx1)'
    )
    parser.add_argument(
        '--pattern',
        type=str,
        default='*.out',
        help='File pattern for batch mode (default: *.out)'
    )

    args = parser.parse_args()

    input_path = Path(args.input)

    if not input_path.exists():
        print(f"Error: {input_path} does not exist")
        sys.exit(1)

    # Mode-specific processing
    if args.mode == 'single':
        if input_path.is_file():
            export_to_csv(str(input_path), args.output, rx_name=args.rx)
        else:
            print("Error: single mode requires a .out file")
            sys.exit(1)

    elif args.mode == 'batch':
        if input_path.is_dir():
            export_batch(str(input_path), args.output, pattern=args.pattern)
        else:
            print("Error: batch mode requires a directory")
            sys.exit(1)

    elif args.mode == 'sequence':
        if input_path.is_dir():
            out_files = sorted(input_path.glob(args.pattern))
            if not out_files:
                print(f"Error: No .out files found in {input_path}")
                sys.exit(1)

            export_sequence_to_single_csv(
                [str(f) for f in out_files],
                args.output,
                component=args.component,
                rx_name=args.rx
            )
        else:
            print("Error: sequence mode requires a directory")
            sys.exit(1)


if __name__ == '__main__':
    main()
