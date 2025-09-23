import boto3

def connect_s3():
    print("S3")
    return boto3.client("s3")

def connect_sqs():
    print("SQS")
    return boto3.client("sqs")


if __name__ == "__main__":
    def connect_s3():
        print("mock S3")
        return None

    def connect_sqs():
        print("mock SQS")
        return None
