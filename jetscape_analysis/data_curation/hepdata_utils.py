"""Utilities for working with HEPdata files.

.. codeauthor:: Raymond Ehlers <raymond.ehlers@cern.ch>, LBL/UCB
"""

from __future__ import annotations

import logging
import re
import tarfile
from pathlib import Path
from typing import TYPE_CHECKING, Any, Final

import attrs
import requests
import ruamel.yaml

if TYPE_CHECKING:
    from jetscape_analysis.data_curation import observable


logger = logging.getLogger(__name__)

_here = Path(__file__).parent
# TODO(RJE): Update this value once I switch to the new repo!
# base_data_dir = _here.parent.parent / "data" / "STAT"
base_data_dir = _here.parent.parent.parent / "hard-sector-data-curation"
DEFAULT_DATABASE_NAME: Final[Path] = Path("hepdata_database.yaml")


def hepdata_filename_from_parameters(record_id: int, version: int | None = None):
    if version is not None and version != -1:
        return f"HEPData-ins{record_id}-v{version}-yaml"
    return f"HEPData-ins{record_id}-yaml"


def parse_content_disposition(content_disposition: str) -> str | None:
    """Parse out content-disposition header to determine the filename."""
    filename = None
    params = content_disposition.split(";")
    for param in params:
        param = param.strip()  # noqa: PLW2901
        match = re.match(r'filename\s*=\s*"?([^"]*)"?', param, re.IGNORECASE)
        if match:
            filename = match.group(1)
            break
    return filename


def extract_record_id_from_hepdata_url(url: str) -> tuple[str, int | None]:
    """Extract record ID from HEPData url.

    Note:
        We don't allow for version specification in the URL - we just take the latest version.
    """
    if "ins" not in url:
        msg = "Invalid HEPData URL format"
        raise ValueError(msg)
    possible_record_id = url.split("ins")[-1].split("/")[0]

    if "v" in possible_record_id:
        # Version was specified. We actually don't want that here, so raise as such.
        msg = "It appears that there's a version specified in the HEPData url. We don't support this (we always take the latest) - please update."
        raise ValueError(msg)

    return possible_record_id


def download_hepdata(
    record_id: int, output_file_path: Path, version: int | None = None, base_dir: Path | None = None
) -> Path:
    """Download HEPData file.

    Args:
        record_id: InspireHEP ID, which is also used to identify the HEPData entry.
        output_file_path: Path to where the downloaded file should be stored.
        version: HEPData version. Default: None, which corresponds to the latest version.
        base_dir: Base directory for data. The output path will be: `base_dir/output_file_path/filename`. Default: None,
            which will correspond to the `data` directory in the repository.
    """
    # Validation
    # Grab the record_id (and validate the url)
    # record_id = extract_record_id_from_hepdata_url(hepdata_url)
    # And the paths
    output_file_path = Path(output_file_path)
    base_dir = base_data_dir if base_dir is None else Path(base_dir)

    # api_url = f"https://www.hepdata.net/download/submission/ins{record_id}/yaml"
    api_url = f"https://www.hepdata.net/record/ins{record_id}"

    # Specify the request format. The options are documented here: https://www.hepdata.net/formats
    #
    # We'll always want to download the full YAML archive.
    options = ["format=yaml"]
    # And we can also specify a version if we need to
    if version is not None:
        # But we can specify a version if we really need...
        options.append(f"version={version}")
    if options:
        api_url += "?" + "&".join(options)
    logger.debug(f"Accessing '{api_url}'...")

    # Download the data
    response = requests.get(api_url, stream=True)
    # Raise an exceptions, as needed
    response.raise_for_status()

    # Get filename from headers
    # NOTE: If we can't get the filename, we're unable to determine the version until we read the
    #       submission.yaml. So as the fallback, we'll just take it without the version.
    filename = f"{hepdata_filename_from_parameters(record_id=record_id, version=version)}.tar.gz"
    content_disposition = response.headers.get("Content-Disposition")
    if content_disposition:
        # Only set if we get something meaningful. Otherwise, go with the fallback
        res = parse_content_disposition(content_disposition)
        if res:
            filename = res

    # Define the path to download files to.
    file_path = base_dir / output_file_path / filename
    # import IPython; IPython.embed()

    file_path.parent.mkdir(exist_ok=True, parents=True)
    with file_path.open("wb") as f:
        for chunk in response.iter_content(chunk_size=8192):
            f.write(chunk)
    logger.debug(f"Successfully download file to {file_path=}")

    # And then retrieve the version from HEPData using json too, just so we can ensure we're being consistent.
    api_url = f"https://www.hepdata.net/record/ins{record_id}"
    options = ["format=json", "light=true"]
    if version is not None:
        # But we can specify a version if we really need...
        options.append(f"version={version}")
    if options:
        api_url += "?" + "&".join(options)
    response = requests.get(api_url, stream=True)
    # Raise an exceptions, as needed
    response.raise_for_status()
    # And then retrieve the version
    data = response.json()
    extracted_version = data.get("version")

    return (file_path, extracted_version)


def extract_archive(file_path: Path) -> Path:
    """Extract files from an .tar.gz archive."""
    # Extract the downloaded file to the directory
    with tarfile.open(file_path, "r:gz") as f_tar:
        logger.warning(file_path.parent)
        f_tar.extractall(file_path.parent)
    logger.warning(f"Successfully extracted {file_path=}")

    # Return the path without the `.tar.gz`, which is where the files
    # are stored by convention.
    return file_path.parent / file_path.name.split(".")[0]


def read_metadata(hepdata_dir: Path) -> tuple[int, dict[str, Path]]:
    # Validation
    hepdata_dir = Path(hepdata_dir)

    # Setup
    y = ruamel.yaml.YAML()

    # Load the metadata from the submission file
    submission_file = hepdata_dir / "submission.yaml"
    with submission_file.open() as f:
        all_metadata = list(y.load_all(f))

    # Extract information from the metadata
    # Grab the version form the DOI. It will look something like: "10.17182/hepdata.77101.v2",
    # where 10.17182 seems to somehow refer to HEPData, 77101 is an internal HEPData id (not the inspire HEP record_ID,
    # which we predominately use), and then the version is encoded in the last string. The version is extracted as:
    # - [-1] grabs the entry around the version (it will be there even for v1)
    # - [1:] removes the "v" from the string
    version = int(all_metadata[0]["hepdata_doi"].split(".")[-1][1:])
    # NOTE: We create this map from [1:] because the first entry is solely metadata - nothing to do with a table
    table_name_to_filename_map = {m["name"]: Path(m["data_file"]) for m in all_metadata[1:]}

    return version, table_name_to_filename_map


@attrs.define
class HEPDataInfo:
    directory: Path
    inspire_hep_record_id: int
    version: int
    tables_to_filenames: dict[str, Path]

    def encode(self) -> dict[str, Any]:
        return {
            "directory": str(self.directory),
            "inspire_hep_record_id": self.inspire_hep_record_id,
            "version": self.version,
            "tables_to_filenames": {k: str(v) for k, v in self.tables_to_filenames.items()},
        }

    @classmethod
    def decode(cls, values: dict[str, Any]) -> HEPDataInfo:
        return cls(
            directory=values["directory"],
            inspire_hep_record_id=values["inspire_hep_record_id"],
            version=values["version"],
            tables_to_filenames={k: Path(v) for k, v in values["tables_to_filenames"].items()},
        )


def read_database() -> dict[str, list[HEPDataInfo]]:
    y = ruamel.yaml.YAML()
    database_filename = base_data_dir / DEFAULT_DATABASE_NAME
    with database_filename.open() as f:
        hepdata_database = y.load(f)

    # If the file is empty, it will return back as None, so we protected against this
    if hepdata_database is None:
        hepdata_database = {}

    # Decode into the objects
    hepdata_database = {k: [HEPDataInfo.decode(_v) for _v in v] for k, v in hepdata_database.items()}
    return hepdata_database  # noqa: RET504


def write_info_to_database(
    entries_to_write: dict[str, list[HEPDataInfo]],
    database_filename: Path | None = None,
) -> bool:
    """Write HEPData info to the database for observable(s).

    NOTE:
        Using a list of HEPDataInfo entries to allow for multiple HEPData files to be relevant for a single observable.
    """
    # Validation
    database_filename = Path(DEFAULT_DATABASE_NAME) if database_filename is None else Path(database_filename)
    database_filename = base_data_dir / database_filename
    # Encode the values
    entries_to_write = {k: [_v.encode() for _v in v] for k, v in entries_to_write.items()}

    y = ruamel.yaml.YAML()
    with database_filename.open() as f:
        database: dict[str, Any] = y.load(f)

    # Validation (can happen if the file is totally empty)
    if database is None:
        database = {}
    # Append entries, sort, and write back
    database.update(entries_to_write)
    database = dict(sorted(database.items()))
    with database_filename.open("w") as f:
        y.dump(database, f)

    return True


def retrieve_observable_hepdata(
    observable_str_as_path: Path, inspire_hep_record_id: int, version: int | None = None
) -> HEPDataInfo:
    # First check if it's available - we'll check with the database
    hepdata_database = read_database()
    all_hepdata_info = hepdata_database.get(str(observable_str_as_path))
    if all_hepdata_info:
        # Check that the archives and the output directories exist. If not, then we're probably missing data and need to download it
        locations_to_check = [
            base_data_dir
            / Path("hepdata")
            / observable_str_as_path
            / f"{hepdata_filename_from_parameters(record_id=v.inspire_hep_record_id, version=v.version)}.tar.gz"
            for v in all_hepdata_info
        ]
        # locations_to_check.extend(
        #    [_v.directory for _v in all_hepdata_info]
        # )
        locations_exist = [_v.exists() for _v in locations_to_check]

        if all(locations_exist):
            logger.warning("All files exist - returning hepdata_info.")
            return all_hepdata_info
        logger.warning(f"Missing files: {locations_to_check} - proceeding to download them.")

    # If not available, then it should be downloaded and extracted
    archive_output_path, extracted_version = download_hepdata(
        record_id=inspire_hep_record_id, output_file_path=Path("hepdata") / observable_str_as_path, version=version
    )
    yaml_output_path = extract_archive(file_path=archive_output_path)
    # Extract info
    extracted_version_from_metadata, table_name_to_filename_map = read_metadata(yaml_output_path)

    # Validation on what we've downloaded.
    if version is not None and version != -1 and extracted_version != version:  # noqa: PLR1714
        msg = (
            f"Mismatch between extracted_version ({extracted_version}) and requested version ({version}). Please check"
        )
        raise ValueError(msg)
    if extracted_version != extracted_version_from_metadata:
        msg = f"Extracted version ({extracted_version}) does not match version from metadata ({extracted_version_from_metadata})!"
        raise ValueError(msg)
    # Ensure we always take the right version from here...
    version = extracted_version

    # And write it to the hepdata database
    hepdata_info = HEPDataInfo(
        directory=yaml_output_path.relative_to(base_data_dir),
        inspire_hep_record_id=inspire_hep_record_id,
        version=version,
        tables_to_filenames=table_name_to_filename_map,
    )
    write_info_to_database(entries_to_write={str(observable_str_as_path): [hepdata_info]})

    # And then it can be provided.
    return [hepdata_info]


def retrieve_hepdata(obs: observable.Observable) -> ...:
    record_id, version = obs.inspire_hep_identifier()

    observable_str_as_path = Path(obs.sqrt_s) / obs.observable_class / obs.name

    retrieve_observable_hepdata(
        observable_str_as_path=observable_str_as_path, inspire_hep_record_id=record_id, version=version
    )
