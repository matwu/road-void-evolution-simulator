"""
Visualization of gprMax output files (.out)
"""
import sys
import argparse
from pathlib import Path
import h5py
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm


def load_gpr_output(output_file: str) -> dict:
    """
    Load gprMax output file

    Args:
        output_file: Path to .out file

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


def plot_ascan(data: dict, rx_name: str = 'rx1', component: str = 'Ez',
               output_file: str = None, title: str = None):
    """
    A-scan plot (single trace)

    Args:
        data: GPR data dictionary
        rx_name: Receiver name
        component: Electric/magnetic field component
        output_file: Output filename
        title: Plot title
    """
    if rx_name not in data:
        print(f"Error: Receiver {rx_name} not found")
        return

    rx_data = data[rx_name]

    if component not in rx_data:
        print(f"Error: Component {component} not found")
        available = list(rx_data.keys())
        print(f"Available components: {available}")
        return

    signal = rx_data[component]

    # Time axis
    if 'dt' in data:
        dt = data['dt']
        time = np.arange(len(signal)) * dt * 1e9  # ns
        xlabel = 'Time (ns)'
    else:
        time = np.arange(len(signal))
        xlabel = 'Sample'

    # Plot
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(time, signal, linewidth=0.8)
    ax.set_xlabel(xlabel, fontsize=12)
    ax.set_ylabel(f'{component} Field (V/m or A/m)', fontsize=12)
    ax.set_title(title or f'A-scan: {rx_name} - {component}', fontsize=14)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if output_file:
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"Saved to {output_file}")
    else:
        plt.show()

    plt.close()


def plot_bscan(output_files: list, component: str = 'Ez',
               output_file: str = None, title: str = None):
    """
    B-scan plot (multiple traces)

    Args:
        output_files: List of .out files
        component: Electric/magnetic field component
        output_file: Output filename
        title: Plot title
    """
    traces = []

    for out_file in output_files:
        data = load_gpr_output(out_file)

        # Get first receiver data
        rx_name = list(data.keys())[0] if data else None
        if rx_name and rx_name.startswith('rx'):
            if component in data[rx_name]:
                traces.append(data[rx_name][component])

    if not traces:
        print("Error: No data found")
        return

    # Create B-scan data
    bscan = np.array(traces)

    # Plot
    fig, ax = plt.subplots(figsize=(14, 8))

    # Colormap
    im = ax.imshow(
        bscan.T,
        aspect='auto',
        cmap='seismic',
        interpolation='bilinear',
        extent=[0, len(traces), bscan.shape[1], 0]
    )

    ax.set_xlabel('Trace Number', fontsize=12)
    ax.set_ylabel('Time Sample', fontsize=12)
    ax.set_title(title or f'B-scan: {component}', fontsize=14)

    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label(f'{component} Field Amplitude', fontsize=11)

    plt.tight_layout()

    if output_file:
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"Saved to {output_file}")
    else:
        plt.show()

    plt.close()


def plot_all_components(output_file: str, rx_name: str = 'rx1',
                        save_dir: str = None):
    """
    Plot all components at once

    Args:
        output_file: .out file path
        rx_name: Receiver name
        save_dir: Save directory
    """
    data = load_gpr_output(output_file)

    if rx_name not in data:
        print(f"Error: Receiver {rx_name} not found")
        return

    rx_data = data[rx_name]
    components = [c for c in rx_data.keys() if c in ['Ex', 'Ey', 'Ez', 'Hx', 'Hy', 'Hz']]

    if not components:
        print("Error: No field components found")
        return

    # Time axis
    if 'dt' in data:
        dt = data['dt']
        n_samples = len(rx_data[components[0]])
        time = np.arange(n_samples) * dt * 1e9  # ns
        xlabel = 'Time (ns)'
    else:
        time = np.arange(len(rx_data[components[0]]))
        xlabel = 'Sample'

    # Subplots
    n_components = len(components)
    fig, axes = plt.subplots(n_components, 1, figsize=(12, 3*n_components))

    if n_components == 1:
        axes = [axes]

    for i, component in enumerate(components):
        signal = rx_data[component]

        axes[i].plot(time, signal, linewidth=0.8)
        axes[i].set_ylabel(f'{component}', fontsize=11)
        axes[i].grid(True, alpha=0.3)

        if i == len(components) - 1:
            axes[i].set_xlabel(xlabel, fontsize=11)

    base_name = Path(output_file).stem
    fig.suptitle(f'GPR Output: {base_name}', fontsize=14, y=0.995)
    plt.tight_layout()

    if save_dir:
        save_path = Path(save_dir) / f'{base_name}_all_components.png'
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved to {save_path}")
    else:
        plt.show()

    plt.close()


def main():
    """Main execution"""
    parser = argparse.ArgumentParser(description='Visualize gprMax output files')
    parser.add_argument(
        'input',
        type=str,
        help='Input .out file or directory containing .out files'
    )
    parser.add_argument(
        '--mode',
        type=str,
        default='ascan',
        choices=['ascan', 'bscan', 'all'],
        help='Plot mode: ascan (single trace), bscan (multiple traces), all (all components)'
    )
    parser.add_argument(
        '--component',
        type=str,
        default='Ez',
        choices=['Ex', 'Ey', 'Ez', 'Hx', 'Hy', 'Hz'],
        help='Field component to plot'
    )
    parser.add_argument(
        '--rx',
        type=str,
        default='rx1',
        help='Receiver name (default: rx1)'
    )
    parser.add_argument(
        '--output',
        type=str,
        help='Output image file path'
    )
    parser.add_argument(
        '--title',
        type=str,
        help='Plot title'
    )

    args = parser.parse_args()

    input_path = Path(args.input)

    if not input_path.exists():
        print(f"Error: {input_path} does not exist")
        sys.exit(1)

    # Process by mode
    if args.mode == 'ascan':
        if input_path.is_file():
            data = load_gpr_output(str(input_path))
            plot_ascan(
                data,
                rx_name=args.rx,
                component=args.component,
                output_file=args.output,
                title=args.title
            )
        else:
            print("Error: A-scan mode requires a single .out file")
            sys.exit(1)

    elif args.mode == 'bscan':
        if input_path.is_dir():
            out_files = sorted(input_path.glob('*.out'))
            if not out_files:
                print(f"Error: No .out files found in {input_path}")
                sys.exit(1)

            plot_bscan(
                [str(f) for f in out_files],
                component=args.component,
                output_file=args.output,
                title=args.title
            )
        else:
            print("Error: B-scan mode requires a directory containing .out files")
            sys.exit(1)

    elif args.mode == 'all':
        if input_path.is_file():
            save_dir = args.output or str(input_path.parent / 'plots')
            plot_all_components(
                str(input_path),
                rx_name=args.rx,
                save_dir=save_dir
            )
        else:
            print("Error: 'all' mode requires a single .out file")
            sys.exit(1)


if __name__ == '__main__':
    main()
