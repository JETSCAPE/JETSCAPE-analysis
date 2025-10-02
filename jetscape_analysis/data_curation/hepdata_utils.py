"""Utilities for working with HEPdata files.

.. codeauthor:: Raymond Ehlers <raymond.ehlers@cern.ch>, LBL/UCB
"""

from __future__ import annotations

import logging
import re
import tarfile
from pathlib import Path
from typing import TYPE_CHECKING, Any, Final
from urllib.parse import parse_qs, urlparse

import attrs
import requests
import ruamel.yaml

if TYPE_CHECKING:
    from jetscape_analysis.data_curation import observable

logger = logging.getLogger(__name__)

_here = Path(__file__).parent
# TODO(RJE): Update this value once I switch to the new repo! the data curation repo will then be a git submodule
# BASE_DATA_DIR = _here.parent.parent / "data" / "STAT"
BASE_DATA_DIR = _here.parent.parent.parent / "hard-sector-data-curation"
DEFAULT_DATABASE_NAME: Final[Path] = Path("hepdata_database.yaml")


def _parse_content_disposition(content_disposition: str) -> str | None:
    """Parse out content-disposition header to determine the filename.

    Args:
        content_disposition: Content-disposition header from the request.
    Returns:
        Extracted filename (assume it's available), or None otherwise.
    """
    filename = None
    params = content_disposition.split(";")
    for param in params:
        param = param.strip()  # noqa: PLW2901
        match = re.match(r'filename\s*=\s*"?([^"]*)"?', param, re.IGNORECASE)
        if match:
            filename = match.group(1)
            break
    return filename


def _encode_query_params(query_params: dict[str, Any]) -> str:
    """Encode query params to add to a URL.

    NOTE:
        Values are converted to strings.

    Args:
        query_params: Parameters to encode into the URL.
    Returns:
        String to be appended to the URL.
    """
    if query_params:
        return "?" + "&".join([f"{k}={v!s}" for k, v in query_params.items()])
    return ""


def hepdata_filename_from_parameters(record_id: int, version: int | None = None):
    if version != -1:
        return f"HEPData-ins{record_id}-v{version}-yaml"
    return f"HEPData-ins{record_id}-yaml"


URLParams = dict[str, Any]


def extract_info_from_hepdata_url(url: str) -> tuple[int, int, URLParams]:
    """Validate the HEPData url matches out expectations.

    We tend to store the HEPData url in yaml config because it's convenient for the user
    (i.e. they can click on it).

    Args:
        url: HEPData url.
    Returns:
        InspireHEP record id, record version, list of URL parameters if the URL is validated. Raises exceptions otherwise.
    Raises:
        ValueError if the URL is not formatted correctly.
    """
    # We expect the URL to be of the form:
    # https://www.hepdata.net/record/ins{record_id}?option_a=2&option_b=3

    # First, extract the InspireHEP record_id
    # Check that the URL parses as expected - namely, it should have format as above, and shouldn't have
    # e.g. any slashes after the record id (e.g. should not have '/v2/')
    url_split_on_slash = url.split("/")
    if len(url_split_on_slash) != 5:
        msg = f"URL appears to have the wrong number of '/' (provided: {url}). We expect starting with 'https://' and no additional slashes after the 'ins{{record_id}}'"
        raise ValueError(msg)

    # Extract the record_id
    full_record_with_possible_options = url_split_on_slash[-1]
    # Remove the options to get the record - even if there is no options (corresponding to "?"),
    # by taking the first value, we'll get the record_id
    record_id_without_params = full_record_with_possible_options.split("?")[0]
    # And then need to remove "ins" for the URL
    record_id = int(record_id_without_params.split("ins")[-1])

    # Now we can parse out the options.
    #
    # Allowed options include:
    # - version=a
    allowed_options = ["version"]
    # Disallowed options include:
    # - format: We want to set our own format as needed - not allow it to be set by the user
    # - light: This returns a reduced amount of info. Again, we want to set our own options, not allow the user.
    disallowed_options = ["format", "light"]

    # Parse options from the URL
    parsed_url = urlparse(url)
    query_params = parse_qs(parsed_url.query)

    # query_params are of the form {"option_a": ['1'], "option_b": ['3']}.
    # This means that everything is a list (because options can be passed multiple times).
    # However, we don't want to allow that, so we force them to a single value
    try:
        query_params = {k: v[0] for k, v in query_params.items()}
    except KeyError as e:
        msg = f"It appears that multiple values are passed for key {e} parsed from URL '{url}'. Please check it."
        raise ValueError(msg) from e

    # Check for disallowed options
    for k in disallowed_options:
        if k in query_params:
            msg = f"Disallowed parameter {k} found in URL '{url}'. Please fix your URL."
            raise ValueError(msg)

    # Check for additional query params that we don't expect
    # This isn't necessarily a problem - it's just unexpected.
    for k in query_params:
        if k not in allowed_options:
            msg = f"Received unexpected parameter {k}. Will pass them through, but please check if this is intentional"
            logger.warning(msg)

    # Set the types of allowed values
    # We don't want the version to be passed through the query_params - we want to handle it explicitly.
    version = query_params.pop("version", None)
    # Default to -1, which corresponds to be we don't know the version.
    version = int(version) if version else -1

    return record_id, version, query_params


def download_hepdata(
    record_id: int,
    output_file_path: Path,
    version: int | None = None,
    base_dir: Path | None = None,
    additional_query_params: dict[str, Any] | None = None,
) -> tuple[Path, int]:
    """Download HEPData file.

    Args:
        record_id: InspireHEP ID, which is also used to identify the HEPData entry.
        output_file_path: Path to where the downloaded file should be stored.
        version: HEPData version. Default: None, which corresponds to the latest version.
        base_dir: Base directory for data. The output path will be: `base_dir/output_file_path/filename`. Default: None,
            which will correspond to the `data` directory in the repository.
    Returns:
        Path to downloaded file, version extracted from HEPData.
    """
    # Validation
    # First, check the paths
    output_file_path = Path(output_file_path)
    base_dir = BASE_DATA_DIR if base_dir is None else Path(base_dir)
    # Additional query params to add the url
    if not additional_query_params:
        additional_query_params = {}
    if "version" in additional_query_params:
        msg = f"version needs to be passed explicitly - not through the additional_query_params. Provided: {additional_query_params=}"
        raise ValueError(msg)

    # api_url = f"https://www.hepdata.net/download/submission/ins{record_id}/yaml"
    api_url = f"https://www.hepdata.net/record/ins{record_id}"

    # Specify the request format. The options are documented here: https://www.hepdata.net/formats
    #
    # We'll always want to download the full YAML archive.
    # NOTE: We put the additional_query_params first because we need to over
    options = {**additional_query_params, "format": "yaml"}
    # And we can also specify a version if we need to
    if version != -1:
        # But we can specify a version if we really need...
        options["version"] = str(version)
    api_url += _encode_query_params(options)
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
        res = _parse_content_disposition(content_disposition)
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
    options = {**additional_query_params, "format": "json", "light": "true"}
    if version != -1:
        # But we can specify a version if we really need...
        options["version"] = str(version)
    api_url += _encode_query_params(options)
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
        # NOTE: The filter only use the minimal features necessary
        f_tar.extractall(file_path.parent, filter=tarfile.data_filter)
    logger.warning(f"Successfully extracted {file_path=}")

    # Return the path without the `.tar.gz`, which is where the files
    # are stored by convention.
    return file_path.parent / file_path.name.split(".")[0]


def read_metadata_from_HEPData_files(hepdata_dir: Path) -> tuple[int, dict[str, Path]]:
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


@attrs.define(frozen=True)
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
    database_filename = BASE_DATA_DIR / DEFAULT_DATABASE_NAME
    with database_filename.open() as f:
        hepdata_database = y.load(f)

    # If the file is empty, it will return back as None, so we protected against this
    if hepdata_database is None:
        hepdata_database = {}

    # Decode into the objects (better would be to properly register with YAML, but this good enough and easier).
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
    database_filename = BASE_DATA_DIR / database_filename
    # Encode the values (better would be to properly register with YAML, but this good enough and easier).
    entries_to_write = {k: [_v.encode() for _v in v] for k, v in entries_to_write.items()}

    # NOTE: From here, we're working entirely with **ENCODED VALUES**. That is to say, dicts

    # Need to read the database first so we can update it.
    y = ruamel.yaml.YAML()
    with database_filename.open() as f:
        database: dict[str, Any] = y.load(f)

    # Validation (can happen if the file is totally empty)
    if database is None:
        database = {}

    # Add entries, sort, and write back
    # If the record_id is the same, we'll replace the entry. Otherwise, we'll append to the list for the observable.
    # import IPython; IPython.embed()
    logger.warning(f"{entries_to_write=}")
    for k, v in entries_to_write.items():
        logger.warning(f"{v=}")
        existing_info_entries: list[HEPDataInfo] = database.get(k, [])
        # Only keep existing records if they won't be replaced by new records.
        # NOTE: We ignore the version here since we should be safe to assume that anything we're trying
        #       to write is intentional.
        new_record_ids = [_v["inspire_hep_record_id"] for _v in v]
        logger.warning(f"{new_record_ids=}")
        new_info_entries = [_v for _v in existing_info_entries if _v["inspire_hep_record_id"] not in new_record_ids]
        # And the new info entries
        new_info_entries.extend(v)
        # Ensure they're sorted for consistency.
        new_info_entries = sorted(new_info_entries)
        # And finally, store
        database[k] = new_info_entries

    # Finally, keep the database sorted so it's easier to keep track of changes
    database = dict(sorted(database.items()))
    with database_filename.open("w") as f:
        y.dump(database, f)

    return True


def retrieve_observable_hepdata(
    observable_str_as_path: Path,
    inspire_hep_record_id: int,
    version: int = -1,
    base_dir: Path | None = None,
    additional_query_params: dict[str, Any] | None = None,
) -> HEPDataInfo:
    """Retrieve a single HEPData for the observable.

    NOTE:
        An observable may need multiple HEPData files, but they can be loaded
        here one-by-one.

    Args:
        observable_str_as_path: The observable string (e.g. (sqrt_s, observable_class, name)), with each
            value treated as a directory in a path.
        inspire_hep_record_id: Inspire HEP record ID
        version: Version of Inspire HEP record. Default: -1, which corresponds to the newest.
        base_dir: Base directory where the data will be accessed and stored. Default: BASE_DATA_DIR.
        additional_query_params: Additional query parameters to pass when downloading from HEPData. Default: None.
    """
    # Setup and validation
    base_dir = BASE_DATA_DIR if base_dir is None else Path(base_dir)
    # By convention, we keep the data in a directory called "data", so we always
    # want to add it on
    base_dir /= "data"
    # Ensure we have something valid
    if not additional_query_params:
        additional_query_params = {}

    # First, let's check if we already have everything and can just skip ahead.
    # We'll start with the the database
    hepdata_database = read_database()
    all_hepdata_info = hepdata_database.get(str(observable_str_as_path))

    # Checking the database information first
    if all_hepdata_info:
        # Look for the record_id and version in the database
        # NOTE: If the version is -1, we can't actually restrict on the version. So we just take what's there.
        #       If we need to go to a newer version, we must request it explicitly.
        possible_hepdata_info = [
            h
            for h in all_hepdata_info
            if h.inspire_hep_record_id == inspire_hep_record_id and (h.version == version if version != -1 else True)
        ]
        if len(possible_hepdata_info) > 1:
            msg = f"Found multiple records - not sure what to do here. Please take a look: {possible_hepdata_info}"
            raise ValueError(msg)

        if len(possible_hepdata_info) == 1:
            # If we've found it, we'll assume it's okay and just proceed.
            logger.warning(f"Found {inspire_hep_record_id} in database - returning info.")
            return possible_hepdata_info[0]

        # If nothing is found, there's nothing else to be done - we'll just continue with the regular process,
        # which defaults to downloading and extracting files as needed.
        logger.warning("Could not find entry in database. Proceeding to download.")

    # At this point, we need to figure out what operations are required.
    # First, let's check on whether we have .tar.gz archive.
    # If we have the version available, we can just construct the path
    archive_path = (
        base_dir
        / observable_str_as_path
        / hepdata_filename_from_parameters(record_id=inspire_hep_record_id, version=version)
    )
    if version == -1:
        # But if we only passed -1, we don't know the value a priori, so we need to search for it.
        archive_path = list((base_dir / observable_str_as_path).glob(f"*{inspire_hep_record_id}*.tar.gz"))
        # If we find many, then bail out - I'm not sure what to do in that case.
        if len(archive_path) > 1:
            msg = f"Looking for existing archive, but found more than one?? {archive_path}"
            raise ValueError(msg)
        # If there's only one, we've probably found it, so
        if len(archive_path) == 1:
            archive_path = archive_path[0]
        # If we found nothing, then leave the default path as above - it probably won't be found, which is fine

    # This is strictly redundant in the case of version == -1, but it simplifies the flow, so will just go with it.
    archive_exists = archive_path.exists()

    # If not available, then it should be downloaded and extracted
    if not archive_exists:
        archive_path, extracted_version_from_hepdata_website = download_hepdata(
            record_id=inspire_hep_record_id,
            output_file_path=observable_str_as_path,
            version=version,
            base_dir=base_dir,
            additional_query_params=additional_query_params,
        )
    else:
        # We already have the archive, so we can skip on to the next step.
        # We don't have a version to extract from the website, so let's skip it (as -1)
        extracted_version_from_hepdata_website = -1

    # Now, let's check for the directories where we unarchive everything
    # NOTE: Need to use split(".") rather than .stem because stem on `.tar.gz` return `.tar`
    yaml_data_dir = archive_path.parent / archive_path.stem.split(".")[0]
    if not archive_exists or not yaml_data_dir.exists():
        # We already have the path, so we don't need to keep track of the returned value
        extract_archive(file_path=archive_path)

    # Extract info from the submission.yaml file
    extracted_version_from_metadata, table_name_to_filename_map = read_metadata_from_HEPData_files(yaml_data_dir)

    # Validation the versions that we have available.
    # Can only check the requested version vs the one extracted from the hepdata website if they're both specified
    if version != -1 and extracted_version_from_hepdata_website not in [-1, version]:
        msg = f"Mismatch between extracted_version ({extracted_version_from_hepdata_website}) and requested version ({version}). Please check"
        raise ValueError(msg)
    # NOTE: We don't need to check extracted_version_from_metadata != -1 since we're always
    #       have valid metadata and a valid version from the metadata at this point in the function.
    if extracted_version_from_hepdata_website not in [-1, extracted_version_from_metadata]:
        msg = f"Extracted version ({extracted_version_from_hepdata_website}) does not match version from metadata ({extracted_version_from_metadata})!"
        raise ValueError(msg)
    # We've done all the validation we can, so may as well set the
    # Ensure we always take the right version from here...
    version = extracted_version_from_metadata

    # And write it to the hepdata database
    hepdata_info = HEPDataInfo(
        directory=yaml_data_dir.relative_to(base_dir),
        inspire_hep_record_id=inspire_hep_record_id,
        version=version,
        tables_to_filenames=table_name_to_filename_map,
    )
    write_info_to_database(entries_to_write={str(observable_str_as_path): [hepdata_info]})

    # And then it can be provided.
    return hepdata_info


def retrieve_hepdata(obs: observable.Observable) -> ...:
    record_id, version = obs.inspire_hep_identifier()

    observable_str_as_path = Path(obs.sqrt_s) / obs.observable_class / obs.name

    retrieve_observable_hepdata(
        observable_str_as_path=observable_str_as_path, inspire_hep_record_id=record_id, version=version
    )


def main() -> None:
    from jetscape_analysis.base import helpers  # noqa: PLC0415

    helpers.setup_logging(level=logging.INFO)
    for observable_str, url in [
        ("5020/inclusive_chjet/angularity_alice", "https://www.hepdata.net/record/ins2845788"),
        ("5020/hadron/pt_ch_cms", "https://www.hepdata.net/record/ins1496050"),
    ]:
        record_id, version, query_params = extract_info_from_hepdata_url(url)
        logger.info(f"Querying with {record_id=}, {version=}, {query_params=}")
        retrieve_observable_hepdata(Path(observable_str), inspire_hep_record_id=record_id, version=version)


if __name__ == "__main__":
    main()
