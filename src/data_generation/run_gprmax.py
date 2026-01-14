"""
gprMax execution wrapper: Run gprMax directly from Python
"""
import os
import sys
import subprocess
from pathlib import Path
from typing import List, Optional
import multiprocessing as mp
from tqdm import tqdm
import yaml


class GPRMaxRunner:
    """gprMax execution class"""

    def __init__(self, num_workers: int = None):
        """
        Args:
            num_workers: Number of parallel workers (defaults to CPU count if None)
        """
        self.num_workers = num_workers or mp.cpu_count()

    def run_single(
        self,
        input_file: str,
        output_dir: Optional[str] = None,
        gpu: Optional[int] = None,
        geometry_only: bool = False
    ) -> bool:
        """
        Run a single gprMax simulation

        Args:
            input_file: Input file path (.in)
            output_dir: Output directory
            gpu: GPU ID (CPU if None)
            geometry_only: Generate geometry only

        Returns:
            Whether successful or not
        """
        input_path = Path(input_file)
        if not input_path.exists():
            print(f"Error: Input file not found: {input_file}")
            return False

        # Build gprMax command
        cmd = ["python", "-m", "gprMax", str(input_path)]

        if output_dir:
            cmd.extend(["-outputdir", output_dir])

        # gprMax -gpu option: simply add "-gpu" flag
        # Note: gprMax automatically uses the first available GPU
        if gpu is not None:
            cmd.append("-gpu")

        if geometry_only:
            cmd.append("-geometry-only")

        try:
            # Execute
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                check=True
            )

            # Check output file
            if not geometry_only:
                output_file = input_path.with_suffix('.out')
                if output_dir:
                    output_file = Path(output_dir) / output_file.name

                if output_file.exists():
                    return True
                else:
                    print(f"Warning: Output file not created: {output_file}")
                    return False
            else:
                return True

        except subprocess.CalledProcessError as e:
            print(f"Error running gprMax for {input_file}:")
            print(e.stderr)
            return False

        except Exception as e:
            print(f"Unexpected error: {e}")
            return False

    def _run_worker(self, args):
        """Worker for parallel execution"""
        input_file, output_dir, gpu = args
        return self.run_single(input_file, output_dir, gpu)

    def run_batch(
        self,
        input_files: List[str],
        output_dir: Optional[str] = None,
        use_gpu: bool = False
    ) -> dict:
        """
        Run multiple gprMax simulations in parallel

        Args:
            input_files: List of input files
            output_dir: Output directory
            use_gpu: Whether to use GPU

        Returns:
            Execution result statistics
        """
        print(f"Running {len(input_files)} simulations with {self.num_workers} workers...")

        # Get GPU count (using pycuda)
        num_gpus = 0
        if use_gpu:
            try:
                import pycuda.driver as drv
                drv.init()
                num_gpus = drv.Device.count()
                if num_gpus > 0:
                    print(f"Found {num_gpus} GPU(s)")
                else:
                    print("Warning: No GPU found, falling back to CPU")
                    use_gpu = False
            except ImportError:
                print("Warning: pycuda not available, falling back to CPU")
                use_gpu = False
            except Exception as e:
                print(f"Warning: GPU detection failed: {e}, falling back to CPU")
                use_gpu = False

        # Prepare arguments for parallel execution
        tasks = []
        for i, input_file in enumerate(input_files):
            # Assign GPU ID when using GPU (rotate if multiple GPUs available)
            gpu = (i % num_gpus) if (use_gpu and num_gpus > 0) else None
            tasks.append((input_file, output_dir, gpu))

        # Parallel execution
        success_count = 0
        failed_files = []

        if self.num_workers > 1:
            with mp.Pool(self.num_workers) as pool:
                results = list(tqdm(
                    pool.imap(self._run_worker, tasks),
                    total=len(tasks),
                    desc="Running gprMax"
                ))

                for input_file, success in zip(input_files, results):
                    if success:
                        success_count += 1
                    else:
                        failed_files.append(input_file)
        else:
            # Single-threaded execution
            for input_file, output_dir, gpu in tqdm(tasks, desc="Running gprMax"):
                success = self.run_single(input_file, output_dir, gpu)
                if success:
                    success_count += 1
                else:
                    failed_files.append(input_file)

        # Result summary
        stats = {
            'total': len(input_files),
            'success': success_count,
            'failed': len(failed_files),
            'failed_files': failed_files
        }

        return stats


def main():
    """Main execution function"""
    import argparse

    parser = argparse.ArgumentParser(description='Run gprMax simulations')
    parser.add_argument(
        '--input-dir',
        type=str,
        default='data/simulations',
        help='Input directory containing .in files'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        help='Output directory (default: same as input)'
    )
    parser.add_argument(
        '--workers',
        type=int,
        default=None,
        help='Number of parallel workers'
    )
    parser.add_argument(
        '--gpu',
        action='store_true',
        help='Use GPU if available'
    )
    parser.add_argument(
        '--pattern',
        type=str,
        default='*.in',
        help='File pattern to match (default: *.in)'
    )

    args = parser.parse_args()

    # Search for input files
    input_dir = Path(args.input_dir)
    if not input_dir.exists():
        print(f"Error: Input directory not found: {input_dir}")
        sys.exit(1)

    input_files = sorted(input_dir.glob(args.pattern))
    input_files = [str(f) for f in input_files]

    if not input_files:
        print(f"Error: No input files found matching pattern: {args.pattern}")
        sys.exit(1)

    print(f"Found {len(input_files)} input files")

    # Run gprMax
    runner = GPRMaxRunner(num_workers=args.workers)
    stats = runner.run_batch(
        input_files,
        output_dir=args.output_dir,
        use_gpu=args.gpu
    )

    # Display results
    print("\n" + "="*50)
    print("SIMULATION RESULTS")
    print("="*50)
    print(f"Total:   {stats['total']}")
    print(f"Success: {stats['success']}")
    print(f"Failed:  {stats['failed']}")

    if stats['failed_files']:
        print("\nFailed files:")
        for f in stats['failed_files']:
            print(f"  - {f}")

    print("="*50)

    # Exit with error if any failures occurred
    if stats['failed'] > 0:
        sys.exit(1)


if __name__ == '__main__':
    main()
