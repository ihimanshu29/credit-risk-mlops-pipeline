from mlProject.constants import * #importing yaml file paths 
from mlProject.utils.common import read_yaml, create_directories
from mlProject.entity.config_entity import (DataIngestionConfig,
                                            DataValidationConfig,
                                            DataTransformationConfig,
                                            ModelTrainerConfig,
                                            ModelEvaluationConfig)




class ConfigurationManager:
    def __init__(
        self,
        #assigning yaml filepaths to thw user variable
        config_filepath = CONFIG_FILE_PATH,
        params_filepath = PARAMS_FILE_PATH,
        schema_filepath = SCHEMA_FILE_PATH):

        #reading yaml paths to locate yaml files
        self.config = read_yaml(config_filepath)
        self.params = read_yaml(params_filepath)
        self.schema = read_yaml(schema_filepath)

        create_directories([self.config.artifacts_root])


    def get_data_ingestion_config(self) -> DataIngestionConfig:
        # here self.config already points to the conig_yaml where we are accessing data_ingestion block and assigning to config
        config = self.config.data_ingestion
        # inside data_ingestion block in yaml "root_dir" key contains value "artifacts/data_ingestion" and we are creating folders with similar setting if not already existing
        create_directories([config.root_dir])
        # here config contains whole data_ingestion block through which we are accessing sub keys like root_dir, source_url etc and storing respective values in the class and also storing them as instance inside data_ingestion_config 
        data_ingestion_config = DataIngestionConfig(
            root_dir=config.root_dir,
            source_URL=config.source_URL,
            local_data_file=config.local_data_file,
            unzip_dir=config.unzip_dir 
        )
        #returning root dir, s url, etc.
        return data_ingestion_config

    
    def get_data_validation_config(self) -> DataValidationConfig:
        config = self.config.data_validation
        schema = self.schema.COLUMNS

        create_directories([config.root_dir])

        data_validation_config = DataValidationConfig(
            root_dir=config.root_dir,
            STATUS_FILE=config.STATUS_FILE,
            unzip_data_dir = config.unzip_data_dir,
            all_schema=schema,
        )

        return data_validation_config


    
    def get_data_transformation_config(self) -> DataTransformationConfig:
        config = self.config.data_transformation

        create_directories([config.root_dir])

        data_transformation_config = DataTransformationConfig(
            root_dir=config.root_dir,
            data_path=config.data_path,
            schema_columns=self.schema.COLUMNS,
            target_column=self.schema.TARGET_COLUMN.name
        )

        return data_transformation_config
    

    
    def get_model_trainer_config(self) -> ModelTrainerConfig:
        config = self.config.model_trainer
        params = self.params.XGBoost
        schema =  self.schema.TARGET_COLUMN

        create_directories([config.root_dir])

        model_trainer_config = ModelTrainerConfig(
            root_dir=config.root_dir,
            train_data_path = config.train_data_path,
            test_data_path = config.test_data_path,
            model_name = config.model_name,
            n_estimators = params.n_estimators,
            max_depth = params.max_depth,
            learning_rate = params.learning_rate,
            scale_pos_weight = params.scale_pos_weight,
            target_column = schema.name
            
        )

        return model_trainer_config


    
    def get_model_evaluation_config(self) -> ModelEvaluationConfig:
        config = self.config.model_evaluation
        params = self.params.XGBoost
        schema =  self.schema.TARGET_COLUMN

        create_directories([config.root_dir])

        model_evaluation_config = ModelEvaluationConfig(
            root_dir=config.root_dir,
            test_data_path=config.test_data_path,
            model_path = config.model_path,
            all_params=params,
            metric_file_name = config.metric_file_name,
            target_column = schema.name
           
        )

        return model_evaluation_config