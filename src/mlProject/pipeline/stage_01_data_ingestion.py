from mlProject.config.configuration import ConfigurationManager
from mlProject.components.data_ingestion import DataIngestion
from mlProject import logger


STAGE_NAME = "Data Ingestion stage"


class DataIngestionTrainingPipeline:
    def __init__(self):
        pass

    def main(self):
        config = ConfigurationManager()
        data_ingestion_config = config.get_data_ingestion_config()      # access root_dir, source_URL, local_data_file, unzip_dir
        data_ingestion = DataIngestion(config=data_ingestion_config) #stores abovw instance(data_ingestion_config) for later use like here inside download file aND EXTRACT ZIP FILE (EXP: self.config.local_data_file)
        data_ingestion.download_file()                              #downloads data.zip from url
        data_ingestion.extract_zip_file()                           #extracts csv file from zip file


if __name__ == '__main__':
    try:
        logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<")
        obj = DataIngestionTrainingPipeline()
        obj.main()
        logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx==========x")
    except Exception as e:
        logger.exception(e)
        raise e