"""Split a merged ascii output file (i.e. with outputs appended) into multiple files.

.. codeauthor:: Raymond Ehlers <raymond.ehlers@cern.ch>, LBL/UCB
"""

import argparse
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


def write_output_file(
    input_filename: Path, current_contents: list[str], file_index: int, n_digits_padding: int = 4
) -> None:
    """Write an output file based on the accumulated file contents.

    Args:
        input_filename: Filename of the input file.
        current_contents: Contents of the input file that we want to write to the new output file.
        file_index: Index to identify the file that we're writing out.
        n_digits_padding: Number of digits to pad the file_index.

    Returns:
        None. The file is written to disk.
    """
    output_filename = (
        input_filename.parent / f"{input_filename.stem}_{file_index:0{n_digits_padding}d}{input_filename.suffix}"
    )

    with output_filename.open("w") as output_file:
        output_file.writelines(current_contents)

    logger.info(f"Created file {output_filename} with {len(current_contents)} lines.")


# This string corresponds to the start of an output file.
_string_to_split_on_by_model = {
    "hybrid": "# event 0",
    "jetscape": "#\tEvent\t1",
}


def split_file_on_string(filename: Path, model_name: str, string_to_split_on: str | None = None) -> bool:
    """Split a text file into new files whenever a key string is encountered.

    This can be used to split apart merged ascii output files which have been
    appended one after another. Each time the key string is encountered, the
    accumulated lines are written to a file, and a new file is started. Note that
    the key string is written to the next file, so no lines are ever removed.

    Args:
        filename: Filename of the file to split apart.
        model_name: Name of the model which generated the file that is being split apart.
            This is used to determine the key string that is used for splittings files
            (unless that key is explicitly specified - see string_to_split_on).
        string_to_split_on: When this string is found, the file should be split
            apart right at the point right BEFORE this string. Default: None,
            which corresponds taking the default based on the provided model.

    Returns:
        True if successful.
    """
    # Validation
    if string_to_split_on is None:
        string_to_split_on = _string_to_split_on_by_model[model_name]

    # Keep track of how where we are
    file_index = 0
    current_contents = []

    try:
        with filename.open() as f:
            for line in f:
                if line.strip().startswith(string_to_split_on):
                    # We've found a new file!
                    # First, save existing output to file
                    # NOTE: If this is empty, it means we just started. In that case, there's nothing to be done.
                    if current_contents:
                        write_output_file(
                            current_contents=current_contents,
                            input_filename=filename,
                            file_index=file_index,
                        )
                        file_index += 1

                    # Restart the list by overwriting the contents, beginning with the new contents
                    current_contents = [line]
                else:
                    # Or just keep recording
                    current_contents.append(line)

            # Write any remaining contents
            if current_contents:
                write_output_file(
                    current_contents=current_contents,
                    input_filename=filename,
                    file_index=file_index,
                )
                file_index += 1

    except OSError as e:
        logger.exception(e)
        logger.error(f"Error reading/writing files: {e}")
        return False
    except Exception as e:
        logger.exception(e)
        logger.error(f"Unexpected error: {e}")
        return False

    return True


def entry_point() -> None:
    """Entry point for splitting appended output files."""
    # Setup logging (delayed important to make it more self contained)
    from jetscape_analysis.base import helpers  # noqa: PLC0415

    helpers.setup_logging(level=logging.DEBUG)

    # Argument parser
    parser = argparse.ArgumentParser(
        description="Split a merged ascii output file (i.e. with outputs appended) into multiple files based on a delimiter string.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "input_file",
        help="Input file to split",
        type=Path,
    )
    parser.add_argument(
        "-m",
        "--model",
        default="hybrid",
        type=str,
        help="Model which generated the file that is being split apart. Options: ['hybrid', 'jetscape']",
    )

    args = parser.parse_args()

    res = split_file_on_string(
        filename=args.input_file,
        model_name=args.model,
    )
    if res:
        logger.info("🎉 Success!")
    else:
        logger.error("❌ Splitting file failed. See log")


if __name__ == "__main__":
    entry_point()
