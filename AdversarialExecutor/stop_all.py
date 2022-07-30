import glob
import json

if __name__ == '__main__':
    executors = glob.glob("./ExecutorConfigs/*.json")

    for e in executors:
        with open(e, "r+") as jsonFile:
            data = json.load(jsonFile)

            data["Allow_TO_RUN"] = False

            jsonFile.seek(0)
            json.dump(data, jsonFile)
            jsonFile.truncate()