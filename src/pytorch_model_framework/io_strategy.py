import os
import subprocess
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional
from urllib.parse import urlparse

import boto3


class InvalidUri(Exception):
    pass


class IoStrategy(ABC):
    def __init__(self, location: str):
        self.location = location

    @abstractmethod
    def save_file(self, file_name: str, contents: str):
        pass

    @abstractmethod
    def file_exists(self) -> bool:
        pass

    @classmethod
    def from_location(cls, location: str) -> "IoStrategy":
        return S3Strategy(location=location) if location.startswith("s3://") else LocalStrategy(location=location)


@dataclass
class S3UriComponents:
    bucket: str
    prefix: str

    @staticmethod
    def from_string(uri: str) -> "S3UriComponents":
        parsed = urlparse(uri, allow_fragments=False)

        components = S3UriComponents(bucket=parsed.netloc, prefix=parsed.path.lstrip("/"))
        if not components.bucket or not components.prefix:
            raise InvalidUri(
                f"Location entered is not valid format: {uri}. Valid format is: s3://some-bucket/some-prefix/"
            )
        return components


class S3Strategy(IoStrategy):
    def __init__(self, location: str):
        super().__init__(location=location)
        self.s3_components = S3UriComponents.from_string(uri=location)
        self.s3 = boto3.resource("s3")

    def file_exists(self) -> bool:
        client = boto3.client("s3")
        try:
            results = client.list_objects_v2(Bucket=self.s3_components.bucket, Prefix=self.s3_components.prefix)
        except Exception:
            return False
        return "Contents" in results

    def save_file(self, file_name: str, contents: str):
        key = f"{self.s3_components.prefix}/{file_name}"
        s3_object = self.s3.Object(bucket_name=self.s3_components.bucket, key=key)
        s3_object.put(Body=contents)

    @staticmethod
    def download_many(source: str, destination: str, is_dryrun: Optional[bool] = False):
        S3UriComponents.from_string(uri=source)
        args = ["aws", "s3", "cp", source, destination, "--recursive", "--quiet"]
        if is_dryrun:
            args.append("--dryrun")
        try:
            print(f"Executing subprocess.run: {args}")
            subprocess.run(args=args, check=True, capture_output=True)
        except subprocess.CalledProcessError as e:
            print(f"Failed to download from S3: {source}")
            raise e

    @staticmethod
    def upload_file(source: str, destination: str, is_dryrun: Optional[bool] = False):
        args = ["aws", "s3", "cp", source, destination]
        if is_dryrun:
            args.append("--dryrun")
        try:
            print(f"Executing subprocess.run: {args}")
            subprocess.run(args=args, check=True, capture_output=True)
        except subprocess.CalledProcessError as e:
            print(f"Failed to upload file from {source} to S3: {destination}")
            raise e

    @staticmethod
    def download_file(source: str, destination: str, is_dryrun: Optional[bool] = False):
        args = ["aws", "s3", "cp", source, destination]
        if is_dryrun:
            args.append("--dryrun")
        try:
            print(f"Executing subprocess.run: {args}")
            subprocess.run(args=args, check=True, capture_output=True)
        except subprocess.CalledProcessError as e:
            print(f"Failed to download from S3: {source}")
            raise e

class LocalStrategy(IoStrategy):
    def __init__(self, location: str):
        super().__init__(location=location)

    def file_exists(self) -> bool:
        return os.path.exists(self.location)

    def save_file(self, file_name: str, contents: str):
        os.makedirs(self.location, exist_ok=True)
        with open(os.path.join(self.location, file_name), "w") as file:
            file.write(contents)
