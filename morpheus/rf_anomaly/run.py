# Copyright (c) 2021-2022, NVIDIA CORPORATION.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import logging
import os

import click

from morpheus.config import Config, ConfigFIL
from morpheus.config import CppConfig
from morpheus.config import PipelineModes
from morpheus.pipeline.linear_pipeline import LinearPipeline
from morpheus.stages.general.monitor_stage import MonitorStage
from morpheus.stages.input.kafka_source_stage import KafkaSourceStage
from morpheus.stages.output.write_to_kafka_stage import WriteToKafkaStage
from morpheus.stages.preprocess.preprocess_fil_stage import PreprocessFILStage
from morpheus.stages.postprocess.serialize_stage import SerializeStage
from morpheus.stages.preprocess.deserialize_stage import DeserializeStage
from morpheus.stages.inference.triton_inference_stage import TritonInferenceStage, TritonInferenceFIL
from morpheus.stages.postprocess.add_classifications_stage import AddClassificationsStage
from morpheus.utils.logger import configure_logging

from splunk_ingest_stage import WriteToSplunkStage


@click.command()
@click.option(
    "--num_threads",
    default=os.cpu_count(),
    type=click.IntRange(min=1),
    help="Number of internal pipeline threads to use",
)
@click.option(
    "--pipeline_batch_size",
    default=5,
    type=click.IntRange(min=1),
    help=("Internal batch size for the pipeline. Can be much larger than the model batch size. "
          "Also used for Kafka consumers"),
)
@click.option(
    "--model_max_batch_size",
    default=32,
    type=click.IntRange(min=1),
    help="Max batch size to use for the model",
)
@click.option(
    "--model_fea_length",
    default=52,
    type=click.IntRange(min=1),
    help="Features length to use for the model",
)
@click.option(
    "--bootstrap_servers",
    default='kafka:9092',
    help="Comma-separated list of bootstrap servers.",
    required=False,
)
@click.option(
    "--input_topic",
    default='in',
    help="Input kafka topic",
    required=False,
)
@click.option(
    "--output_topic",
    default='out',
    help="Output kafka topic",
    required=False,
)


def run_pipeline(
        num_threads,
        pipeline_batch_size,
        model_max_batch_size,
        model_fea_length,
        bootstrap_servers,
        input_topic,
        output_topic,
):
    # Enable the default logger
    configure_logging(log_level=logging.DEBUG)

    CppConfig.set_should_use_cpp(True)

    config = Config()
    config.fil = ConfigFIL()

    # Below properties are specified by the command line
    config.num_threads = num_threads
    config.pipeline_batch_size = pipeline_batch_size
    config.model_max_batch_size = model_max_batch_size
    config.feature_length = model_fea_length
    config.mode = PipelineModes.FIL

    config.class_labels = ['normal', 'anomalous']

    with open('cols.txt', 'r') as cols:
        config.fil.feature_columns = [x.strip() for x in cols.readlines()]

    config.edge_buffer_size = 128

    pipeline = LinearPipeline(config)

    # add doca source stage
    # pipeline.set_source(DocaSourceStage(config, nic_addr, gpu_addr, source_ip_filter))
    # pipeline.set_source(FileSourceStage(config, filename='/workspace/examples/data/pcap_dump.jsonlines', repeat=10))

    pipeline.set_source(KafkaSourceStage(
        config,
        bootstrap_servers=bootstrap_servers,
        input_topic=input_topic
    )
    )

    pipeline.add_stage(MonitorStage(config, description="Ingest rate", unit='messages'))

    pipeline.add_stage(DeserializeStage(config))
    pipeline.add_stage(MonitorStage(config, description="Deserialize rate", unit='messages'))
    pipeline.add_stage(PreprocessFILStage(config))
    pipeline.add_stage(
        TritonInferenceStage(
            config, model_name="rf_anomaly",
            server_url="triton:8000",
            force_convert_inputs=True,
            use_shared_memory=True
        )
    )
    pipeline.add_stage(MonitorStage(config, description="Inference rate", unit='messages'))

    pipeline.add_stage(AddClassificationsStage(config))
    pipeline.add_stage(MonitorStage(config, description="Add Classification rate", unit='messages'))

    pipeline.add_stage(SerializeStage(config))
    pipeline.add_stage(MonitorStage(config, description="Serialization rate", unit='messages'))

    pipeline.add_stage(WriteToKafkaStage(
        config,
        bootstrap_servers=bootstrap_servers,
        output_topic=output_topic
    )
    )

    pipeline.add_stage(MonitorStage(config, description="Output rate", unit='messages'))

    pipeline.build()

    # Run the pipeline
    pipeline.run()


if __name__ == "__main__":
    run_pipeline()
