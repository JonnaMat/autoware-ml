# Copyright 2026 TIER IV, Inc.
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

"""Splitter of dataset records into splits by scenario."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Sequence, Mapping
from types import MappingProxyType

import polars as pl

from autoware_ml.databases.scenarios import Scenarios
from autoware_ml.databases.schemas.dataset_schemas import DatasetRecord, DatasetTableSchema
from autoware_ml.types.dataset import SplitType


class ScenarioSplitter:
    """
    Common Splitter class to split a sequence of DatasetRecord into different splits
    (e.g. train, val, test) based on the given scenarios.

    A database may list no scenario for a split, typically no validation scenario. A
    ``split_fallbacks`` entry then serves that split with the scenarios of another one,
    ``{val: test}`` to validate on the test scenarios. A split that lists scenarios of its own
    never falls back.
    """

    def __init__(self, split_fallbacks: Mapping[str, str] | None = None) -> None:
        """
        Initialize the splitter.

        Args:
          split_fallbacks: Split to the split whose scenarios serve it when it lists none,
            both named by their SplitType value.
        """
        self.split_fallbacks: dict[SplitType, SplitType] = {
            SplitType(split): SplitType(fallback)
            for split, fallback in (split_fallbacks or {}).items()
        }

    def _scenario_ids_by_split(
        self, scenarios: MappingProxyType[str, Scenarios]
    ) -> Mapping[SplitType, set[str]]:
        """
        Collect the scenario IDs of every split, applying the fallbacks of the empty ones.

        Args:
          scenarios: Scenario data of every scenario group.

        Returns:
          Mapping[SplitType, set[str]]: Scenario IDs by split.
        """
        scenario_ids_by_split: Mapping[SplitType, set[str]] = defaultdict(set)
        for scenarios_obj in scenarios.values():
            for split, scenario_data_list in scenarios_obj.scenario_data.items():
                scenario_ids_by_split[split].update(sd.scenario_id for sd in scenario_data_list)
        for split, fallback in self.split_fallbacks.items():
            if not scenario_ids_by_split.get(split):
                scenario_ids_by_split[split] = set(scenario_ids_by_split.get(fallback, set()))
        return scenario_ids_by_split

    def __str__(self) -> str:
        """
        String representation of the splitter.

        Returns:
          str: String representation of the splitter.
        """
        return f"{self.__class__.__name__}()"

    def split_by_dataset_records(
        self,
        dataset_records: Sequence[DatasetRecord],
        scenarios: MappingProxyType[str, Scenarios],
    ) -> MappingProxyType[SplitType, Sequence[DatasetRecord]]:
        """
        Split the dataset records into different splits (e.g. train, val, test) based on the scenarios.

        Args:
          dataset_records: Sequence of dataset records to be split.
          scenarios: MappingProxyType[str, Scenarios] object containing the scenario data for splitting.

        Returns:
          MappingProxyType[SplitType, Sequence[DatasetRecord]]: Mapping from split type to sequence of dataset records in that split.
        """
        scenario_ids_by_split = self._scenario_ids_by_split(scenarios)

        records_by_scenario_id: Mapping[str, list[DatasetRecord]] = defaultdict(list)
        for record in dataset_records:
            records_by_scenario_id[record.scenario_id].append(record)

        split_records: Mapping[SplitType, Sequence[DatasetRecord]] = {}
        for split, scenario_ids in scenario_ids_by_split.items():
            records: Sequence[DatasetRecord] = []
            for scenario_id in scenario_ids:
                records.extend(records_by_scenario_id.get(scenario_id, []))
            split_records[split] = records

        return MappingProxyType(split_records)

    def split_by_polars_dataframe(
        self,
        dataset_records_dataframe: pl.DataFrame,
        scenarios: MappingProxyType[str, Scenarios],
    ) -> MappingProxyType[SplitType, pl.DataFrame]:
        """
        Split the dataset dataframe into different splits (e.g. train, val, test) based on the scenarios.

        Args:
          dataset_records_dataframe: Polars DataFrame containing the dataset records to be split.
          scenarios: MappingProxyType[str, Scenarios] object containing the scenario data for splitting.

        Returns:
          MappingProxyType[SplitType, pl.DataFrame]: Mapping from split type to Polars DataFrame of dataset records in that split.
        """
        scenario_ids_by_split = self._scenario_ids_by_split(scenarios)

        split_dfs: Mapping[SplitType, pl.DataFrame] = {}
        for split, scenario_ids in scenario_ids_by_split.items():
            split_dfs[split] = dataset_records_dataframe.filter(
                pl.col(DatasetTableSchema.SCENARIO_ID.name).is_in(scenario_ids)
            )

        return MappingProxyType(split_dfs)
