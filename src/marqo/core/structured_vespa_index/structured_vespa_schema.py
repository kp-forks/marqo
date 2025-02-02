from typing import List

from marqo.core import constants
from marqo.core.models.marqo_index import FieldType, StructuredMarqoIndex, FieldFeature, Field, TensorField, MarqoIndex
from marqo.core.models.marqo_index_request import StructuredMarqoIndexRequest
from marqo.core.structured_vespa_index import common
from marqo.core.vespa_index.vespa_schema import VespaSchema
from marqo.exceptions import InternalError


class StructuredVespaSchema(VespaSchema):
    _MARQO_TO_VESPA_TYPE_MAP = {
        FieldType.Text: 'string',
        FieldType.Bool: 'byte',
        FieldType.Int: 'int',
        FieldType.Long: 'long',
        FieldType.Float: 'float',
        FieldType.Double: 'double',
        FieldType.ArrayText: 'array<string>',
        FieldType.ArrayInt: 'array<int>',
        FieldType.ArrayLong: 'array<long>',
        FieldType.ArrayFloat: 'array<float>',
        FieldType.ArrayDouble: 'array<double>',
        FieldType.ImagePointer: 'string',
        FieldType.VideoPointer: 'string',
        FieldType.AudioPointer: 'string',
        FieldType.MultimodalCombination: 'map<string, float>',
        FieldType.CustomVector: 'string',  # Custom Vector "content" is stored as string in backend.
        FieldType.MapInt: 'map<string, int>',
        FieldType.MapLong: 'map<string, long>',
        FieldType.MapFloat: 'map<string, float>',
        FieldType.MapDouble: 'map<string, double>'
    }

    _FIELD_INDEX_PREFIX = 'marqo__lexical_'
    _FIELD_FILTER_PREFIX = 'marqo__filter_'
    _FIELD_CHUNKS_PREFIX = 'marqo__chunks_'
    _FIELD_EMBEDDING_PREFIX = 'marqo__embeddings_'

    def __init__(self, index_request: StructuredMarqoIndexRequest):
        self._index_request = index_request

    def generate_schema(self) -> (str, MarqoIndex):
        schema = list()
        schema_name = self._get_vespa_schema_name(self._index_request.name)

        schema.append(f'schema {schema_name} {{')

        document_section, marqo_index = self._generate_document_section(schema_name)

        schema.extend(document_section)
        schema.extend(self._generate_rank_profiles(marqo_index))
        schema.extend(self._generate_default_fieldset(marqo_index))
        schema.extend(self._generate_summaries(marqo_index))
        schema.append('}')

        return '\n'.join(schema), marqo_index

    def _generate_document_section(self, schema_name: str) -> (List[str], StructuredMarqoIndex):
        """
        Generate the document (fields) section of the Vespa schema.

        Returns:
            A tuple of the document section and the StructuredMarqoIndex object.
        """
        document: List[str] = list()
        fields: List[Field] = []
        tensor_fields: List[TensorField] = []

        document.append(f'document {{')

        # ID field
        document.append(f'field {common.FIELD_ID} type string {{')
        document.append('indexing: attribute | summary')
        document.append('attribute: fast-search')
        document.append('rank: filter')
        document.append('}')

        for tensor_field_request in self._index_request.fields:
            field_type = self._get_vespa_type(tensor_field_request.type)
            lexical_field_name = None
            filter_field_name = None

            if FieldFeature.LexicalSearch in tensor_field_request.features:
                field_name = f'{self._FIELD_INDEX_PREFIX}{tensor_field_request.name}'
                document.append(f'field {field_name} type {field_type} {{')
                document.append(f'indexing: index | summary')
                document.append('index: enable-bm25')
                document.append('}')

                lexical_field_name = field_name

            if FieldFeature.Filter in tensor_field_request.features:
                field_name = f'{self._FIELD_FILTER_PREFIX}{tensor_field_request.name}'
                document.append(f'field {field_name} type {field_type} {{')
                document.append('indexing: attribute | summary')
                document.append('attribute: fast-search')
                document.append('rank: filter')
                document.append('}')

                filter_field_name = field_name

            if (
                    FieldFeature.LexicalSearch not in tensor_field_request.features and
                    FieldFeature.Filter not in tensor_field_request.features
            ):
                field_name = tensor_field_request.name
                document.append(f'field {field_name} type {field_type} {{')
                document.append('indexing: summary')
                document.append('}')

            fields.append(
                Field(
                    name=tensor_field_request.name,
                    type=tensor_field_request.type,
                    features=tensor_field_request.features,
                    lexical_field_name=lexical_field_name,
                    filter_field_name=filter_field_name,
                    dependent_fields=tensor_field_request.dependent_fields
                )
            )

        # score modifiers
        if any(FieldFeature.ScoreModifier in f.features for f in self._index_request.fields):
            document.append(
                f'field {common.FIELD_SCORE_MODIFIERS_DOUBLE_LONG} type tensor<double>(p{{}}) {{ indexing: attribute | summary }}',
            )
            document.append(
                f'field {common.FIELD_SCORE_MODIFIERS_FLOAT} type tensor<float>(p{{}}) {{ indexing: attribute | summary }}'
            )

        # tensor fields
        model_dim = self._index_request.model.get_dimension()
        for tensor_field_request in self._index_request.tensor_fields:
            chunks_field_name = f'{self._FIELD_CHUNKS_PREFIX}{tensor_field_request}'
            embedding_field_name = f'{self._FIELD_EMBEDDING_PREFIX}{tensor_field_request}'
            document.append(f'field {chunks_field_name} type array<string> {{')
            document.append('indexing: attribute | summary')
            document.append('}')
            document.append(f'field {embedding_field_name} type tensor<float>(p{{}}, x[{model_dim}]) {{')
            document.append('indexing: attribute | index | summary')
            document.append(
                f'attribute {{ distance-metric: {self._get_distance_metric(self._index_request.distance_metric)} }}')
            document.append('index { hnsw {')
            document.append(f'max-links-per-node: {self._index_request.hnsw_config.m}')
            document.append(f'neighbors-to-explore-at-insert: {self._index_request.hnsw_config.ef_construction}')
            document.append('}}')
            document.append('}')

            tensor_fields.append(
                TensorField(
                    name=tensor_field_request,
                    chunk_field_name=chunks_field_name,
                    embeddings_field_name=embedding_field_name
                )
            )

        # vector count field
        document.append(f'field {common.FIELD_VECTOR_COUNT} type int {{ indexing: attribute | summary }}')

        document.append('}')

        marqo_index = StructuredMarqoIndex(
            name=self._index_request.name,
            schema_name=schema_name,
            model=self._index_request.model,
            normalize_embeddings=self._index_request.normalize_embeddings,
            text_preprocessing=self._index_request.text_preprocessing,
            image_preprocessing=self._index_request.image_preprocessing,
            video_preprocessing=self._index_request.video_preprocessing,
            audio_preprocessing=self._index_request.audio_preprocessing,
            distance_metric=self._index_request.distance_metric,
            vector_numeric_type=self._index_request.vector_numeric_type,
            hnsw_config=self._index_request.hnsw_config,
            marqo_version=self._index_request.marqo_version,
            created_at=self._index_request.created_at,
            updated_at=self._index_request.updated_at,
            fields=fields,
            tensor_fields=tensor_fields,
        )

        return document, marqo_index

    def _generate_max_bm25_expression(self, lexical_fields: List[Field]) -> str:
        """
        Recursively generate max closeness expression for all lexical fields.
        Max is a binary operator, so for more than 2 fields this method gets max of:
        1) first field
        2) max of the rest of the fields.

        prefixed arg determines whether to add "marqo__lexical_" prefix to tensor field name.
        """
        # Base cases
        # If no lexical fields, return empty string.
        # If only 1 or 2 lexical fields, get max of one/both.
        if len(lexical_fields) == 0:
            return ""
        else:
            lexical_field_name = lexical_fields[0].lexical_field_name
            if len(lexical_fields) == 1:
                return (f'if(query({lexical_field_name}) > 0, '
                        f'bm25({lexical_field_name}), 0)')
            # Recursive step
            else:
                return (f'max('
                        f'if(query({lexical_field_name}) > 0, '
                        f'bm25({lexical_field_name}), 0), '
                        f'{self._generate_max_bm25_expression(lexical_fields[1:])})')

    def _generate_max_similarity_expression(self, tensor_fields: List[TensorField]) -> str:
        """
        Recursively generate max closeness expression for all tensor fields.
        Max is a binary operator, so for more than 2 fields this method gets max of:
        1) first field
        2) max of the rest of the fields.

        prefixed arg determines whether to add "marqo__tensor_" prefix to tensor field name.
        """
        # Base cases
        # If no tensor fields, return empty string.
        # If only 1 or 2 tensor fields, get max of one/both.
        if len(tensor_fields) == 0:
            return ""
        else:
            embeddings_field_name = tensor_fields[0].embeddings_field_name
            if len(tensor_fields) == 1:
                return (f'if(query({embeddings_field_name}) > 0, '
                        f'closeness(field, {embeddings_field_name}), 0)')
            # Recursive step
            else:
                return (f'max('
                        f'if(query({embeddings_field_name}) > 0, '
                        f'closeness(field, {embeddings_field_name}), 0), '
                        f'{self._generate_max_similarity_expression(tensor_fields[1:])})')

    def _generate_rank_profiles(self, marqo_index: StructuredMarqoIndex) -> List[str]:
        rank_profiles: List[str] = list()

        lexical_fields = marqo_index.lexical_field_map.values()
        tensor_fields = marqo_index.tensor_fields
        score_modifier_fields_names = marqo_index.score_modifier_fields_names
        model_dim = marqo_index.model.get_dimension()

        # Basic expressions
        bm25_sum_expression = ' + '.join([
            f'if (query({field.lexical_field_name}) > 0, bm25({field.lexical_field_name}), 0)' for field in
            lexical_fields
        ])
        bm25_avg_denominator_sum = ' + '.join([f'bm25({field.lexical_field_name})' for field in lexical_fields])
        bm25_avg_expression = (f'({bm25_sum_expression}) / max(1, ' +
                               ' + '.join([f'if (query({field.lexical_field_name}) > 0, 1, 0)'
                                           for field in lexical_fields]) + ')')
        bm25_max_expression = self._generate_max_bm25_expression(list(lexical_fields))
        embedding_similarity_expression = self._generate_max_similarity_expression(tensor_fields)

        mult_modifier_expression = (
            f'if (count(mult_weights * attribute({common.FIELD_SCORE_MODIFIERS_DOUBLE_LONG})) == 0, '
            f'   1, reduce(mult_weights * attribute({common.FIELD_SCORE_MODIFIERS_DOUBLE_LONG}), prod)) '
            f'* if (count(mult_weights * attribute({common.FIELD_SCORE_MODIFIERS_FLOAT})) == 0, '
            f'   1, reduce(mult_weights * attribute({common.FIELD_SCORE_MODIFIERS_FLOAT}), prod)) '
        ) if score_modifier_fields_names else '1'
        add_modifier_expression = (
            f'reduce(add_weights * attribute({common.FIELD_SCORE_MODIFIERS_DOUBLE_LONG}), sum) '
            f'+ reduce(add_weights * attribute({common.FIELD_SCORE_MODIFIERS_FLOAT}), sum)'
        ) if score_modifier_fields_names else '0'
        score_modifier_expression = (
            'mult_modifier(mult_weights) * score + add_modifier(add_weights)'
        ) if score_modifier_fields_names else 'score'

        embedding_match_features_expression = [f'match-features inherits {common.RANK_PROFILE_BASE} {{']
        for field in marqo_index.tensor_fields:
            embedding_match_features_expression.append(f'closest({field.embeddings_field_name})')
            embedding_match_features_expression.append(f'distance(field, {field.embeddings_field_name})')
        embedding_match_features_expression.append('}')

        # Base rank profile
        rank_profiles.append(f'rank-profile {common.RANK_PROFILE_BASE} inherits default {{')
        rank_profiles.append('inputs {')

        for field in lexical_fields:
            rank_profiles.append(f'query({field.lexical_field_name}): 0')
        for field in tensor_fields:
            rank_profiles.append(f'query({field.embeddings_field_name}): 0')
        rank_profiles.append(f'query({common.QUERY_INPUT_EMBEDDING}) tensor<float>(x[{model_dim}])')
        rank_profiles.append(f'query({common.QUERY_INPUT_BM25_AGGREGATOR}): 0')
        rank_profiles.append(f'query({constants.QUERY_INPUT_SCORE_MODIFIERS_MULT_WEIGHTS_LEXICAL}) tensor<double>(p{{}})')
        rank_profiles.append(f'query({constants.QUERY_INPUT_SCORE_MODIFIERS_ADD_WEIGHTS_LEXICAL}) tensor<double>(p{{}})')
        rank_profiles.append(f'query({constants.QUERY_INPUT_SCORE_MODIFIERS_MULT_WEIGHTS_TENSOR}) tensor<double>(p{{}})')
        rank_profiles.append(f'query({constants.QUERY_INPUT_SCORE_MODIFIERS_ADD_WEIGHTS_TENSOR}) tensor<double>(p{{}})')
        rank_profiles.append(f'query({constants.QUERY_INPUT_SCORE_MODIFIERS_MULT_WEIGHTS_GLOBAL}) tensor<double>(p{{}})')
        rank_profiles.append(f'query({constants.QUERY_INPUT_SCORE_MODIFIERS_ADD_WEIGHTS_GLOBAL}) tensor<double>(p{{}})')

        rank_profiles.append('}')

        rank_profiles.append('function mult_modifier(mult_weights) {')
        rank_profiles.append(f'   expression: {mult_modifier_expression}')
        rank_profiles.append('}')

        rank_profiles.append('function add_modifier(add_weights) {')
        rank_profiles.append(f'   expression: {add_modifier_expression}')
        rank_profiles.append('}')

        rank_profiles.append('function modify(score, mult_weights, add_weights) {')
        rank_profiles.append(f'   expression: {score_modifier_expression}')
        rank_profiles.append('}')

        rank_profiles.append('function global_mult_modifier() {')
        rank_profiles.append(f'   expression: mult_modifier(query({constants.QUERY_INPUT_SCORE_MODIFIERS_MULT_WEIGHTS_GLOBAL}))')
        rank_profiles.append('}')

        rank_profiles.append('function global_add_modifier() {')
        rank_profiles.append(f'   expression: add_modifier(query({constants.QUERY_INPUT_SCORE_MODIFIERS_ADD_WEIGHTS_GLOBAL}))')
        rank_profiles.append('}')

        if lexical_fields:
            rank_profiles.append('function lexical_score_sum() {')
            rank_profiles.append(f'expression: {bm25_sum_expression}')
            rank_profiles.append('}')
            rank_profiles.append('function lexical_score_avg() {')
            rank_profiles.append(f'expression: {bm25_avg_expression}')
            rank_profiles.append('}')
            rank_profiles.append('function lexical_score_max() {')
            rank_profiles.append(f'expression: {bm25_max_expression}')
            rank_profiles.append('}')
            # Decide which of the 3 lexical aggregate functions to use
            rank_profiles.append('function lexical_score() {')
            rank_profiles.append(
                f'expression: if (query({common.QUERY_INPUT_BM25_AGGREGATOR}) == 0, lexical_score_sum(), '
                f'if (query({common.QUERY_INPUT_BM25_AGGREGATOR}) == 1, lexical_score_avg(), '
                f'lexical_score_max()))')
            rank_profiles.append('}')
        if tensor_fields:
            rank_profiles.append('function embedding_score() {')
            rank_profiles.append(f'expression: {embedding_similarity_expression}')
            rank_profiles.append('}')

        # Global add and mult modifiers should be accessible in searcher
        rank_profiles.append('match-features: global_mult_modifier global_add_modifier')

        rank_profiles.append('}')

        if lexical_fields:
            rank_profiles.append(f'rank-profile {common.RANK_PROFILE_BM25} inherits {common.RANK_PROFILE_BASE} {{')

            rank_profiles.append('first-phase {')
            rank_profiles.append(
                f'expression: modify(lexical_score(), '
                f'query({constants.QUERY_INPUT_SCORE_MODIFIERS_MULT_WEIGHTS_LEXICAL}), '
                f'query({constants.QUERY_INPUT_SCORE_MODIFIERS_ADD_WEIGHTS_LEXICAL})'
                f')'
            )
            rank_profiles.append('}}')

        if tensor_fields:
            rank_profiles.append(f'rank-profile {common.RANK_PROFILE_EMBEDDING_SIMILARITY} '
                                 f'inherits {common.RANK_PROFILE_BASE} {{')

            rank_profiles.append('first-phase {')
            rank_profiles.append(f'expression: modify(embedding_score(), '
                                 f'query({constants.QUERY_INPUT_SCORE_MODIFIERS_MULT_WEIGHTS_TENSOR}), '
                                 f'query({constants.QUERY_INPUT_SCORE_MODIFIERS_ADD_WEIGHTS_TENSOR})'
                                 f')')
            rank_profiles.append('}')
            rank_profiles.extend(embedding_match_features_expression)   # This is a multi-line list, so we extend
            rank_profiles.append('}')

        # Hybrid search
        if lexical_fields and tensor_fields:
            rank_profiles.append(f'rank-profile {common.RANK_PROFILE_HYBRID_CUSTOM_SEARCHER} inherits default {{')

            # Input parameters (rank features)
            rank_profiles.append('inputs {')
            rank_profiles.append(f'query({common.QUERY_INPUT_EMBEDDING}) tensor<float>(x[{model_dim}])')
            # for field in tensor_fields:
            #     rank_profiles.append(f'query({field.emb}): 0')

            # Temp parameters to pass into respective queries (lexical and tensor)
            rank_profiles.append(f'query({common.QUERY_INPUT_HYBRID_FIELDS_TO_RANK_LEXICAL}) tensor<int8>(p{{}})')
            rank_profiles.append(f'query({common.QUERY_INPUT_HYBRID_FIELDS_TO_RANK_TENSOR}) tensor<int8>(p{{}})')
            rank_profiles.append(
                f'query({constants.QUERY_INPUT_SCORE_MODIFIERS_MULT_WEIGHTS_LEXICAL}) tensor<double>(p{{}})')
            rank_profiles.append(
                f'query({constants.QUERY_INPUT_SCORE_MODIFIERS_ADD_WEIGHTS_LEXICAL}) tensor<double>(p{{}})')
            rank_profiles.append(
                f'query({constants.QUERY_INPUT_SCORE_MODIFIERS_MULT_WEIGHTS_TENSOR}) tensor<double>(p{{}})')
            rank_profiles.append(
                f'query({constants.QUERY_INPUT_SCORE_MODIFIERS_ADD_WEIGHTS_TENSOR}) tensor<double>(p{{}})')
            rank_profiles.append(
                f'query({constants.QUERY_INPUT_SCORE_MODIFIERS_MULT_WEIGHTS_GLOBAL}) tensor<double>(p{{}})')
            rank_profiles.append(
                f'query({constants.QUERY_INPUT_SCORE_MODIFIERS_ADD_WEIGHTS_GLOBAL}) tensor<double>(p{{}})')

            rank_profiles.append('}')
            rank_profiles.append('}')

            # For hybrid bm25 and embedding_similarity rank profiles
            # HYBRID SEARCH LEXICAL THEN TENSOR
            rank_profiles.append(
                f'rank-profile {common.RANK_PROFILE_HYBRID_BM25_THEN_EMBEDDING_SIMILARITY} inherits '
                f'{common.RANK_PROFILE_BASE} {{'
            )

            rank_profiles.append('first-phase {')  # First phase ranks on LEXICAL
            rank_profiles.append(
                f'expression: modify(lexical_score(), '
                f'query({constants.QUERY_INPUT_SCORE_MODIFIERS_MULT_WEIGHTS_LEXICAL}), '
                f'query({constants.QUERY_INPUT_SCORE_MODIFIERS_ADD_WEIGHTS_LEXICAL}))'
            )
            rank_profiles.append('}')
            rank_profiles.append('second-phase {')  # Second phase ranks on TENSOR
            rank_profiles.append(
                f'expression: modify(embedding_score(), '
                f'query({constants.QUERY_INPUT_SCORE_MODIFIERS_MULT_WEIGHTS_TENSOR}), '
                f'query({constants.QUERY_INPUT_SCORE_MODIFIERS_ADD_WEIGHTS_TENSOR}))'
            )
            rank_profiles.append('}')
            rank_profiles.extend(embedding_match_features_expression)   # This is a multi-line list, so we extend
            rank_profiles.append('}')

            # HYBRID SEARCH TENSOR THEN LEXICAL
            # Only 1 phase ranking (lexical only)
            rank_profiles.append(
                f'rank-profile {common.RANK_PROFILE_HYBRID_EMBEDDING_SIMILARITY_THEN_BM25} inherits '
                f'{common.RANK_PROFILE_BASE} {{'
            )
            rank_profiles.append('first-phase {')  # First phase ranks on LEXICAL
            rank_profiles.append(
                f'expression: modify(lexical_score(), '
                f'query({constants.QUERY_INPUT_SCORE_MODIFIERS_MULT_WEIGHTS_LEXICAL}), '
                f'query({constants.QUERY_INPUT_SCORE_MODIFIERS_ADD_WEIGHTS_LEXICAL}))')
            rank_profiles.append('}}')

        return rank_profiles

    def _generate_default_fieldset(self, marqo_index: StructuredMarqoIndex) -> List[str]:
        fieldsets: List[str] = list()

        fieldset_fields = marqo_index.lexical_field_map.keys()

        if fieldset_fields:
            fieldsets.append('fieldset default {')
            if fieldset_fields:
                fieldsets.append(f'fields: {", ".join(fieldset_fields)}')
            fieldsets.append('}')

        return fieldsets

    def _generate_summaries(self, marqo_index: StructuredMarqoIndex) -> List[str]:
        summaries: List[str] = list()

        non_vector_summary_fields = []
        vector_summary_fields = []

        non_vector_summary_fields.append(
            f'summary {common.FIELD_ID} type string {{ }}'
        )

        for field in marqo_index.fields:
            target_field_name = field.name
            field_type = self._get_vespa_type(field.type)

            if field.type == FieldType.MultimodalCombination:
                # return combination weights only for vector summary
                vector_summary_fields.append(
                    f'summary {target_field_name} type {field_type} {{ }}'
                )
                continue

            if field.filter_field_name:
                # Filter fields are in-memory attributes so use this even if there's a lexical field
                source_field_name = field.filter_field_name
            elif field.lexical_field_name:
                source_field_name = field.lexical_field_name
            else:
                source_field_name = field.name

            non_vector_summary_fields.append(
                f'summary {target_field_name} type {field_type} {{ source: {source_field_name} }}'
            )

        for field in marqo_index.tensor_fields:
            non_vector_summary_fields.append(
                f'summary {field.chunk_field_name} type array<string> {{ }}'
            )
            vector_summary_fields.append(
                f'summary {field.embeddings_field_name} type tensor<float>(p{{}}, '
                f'x[{marqo_index.model.get_dimension()}]) {{ }}'
            )

        summaries.append(f'document-summary {common.SUMMARY_ALL_NON_VECTOR} {{')
        summaries.extend(non_vector_summary_fields)
        summaries.append('}')
        summaries.append(f'document-summary {common.SUMMARY_ALL_VECTOR} {{')
        summaries.extend(non_vector_summary_fields)
        summaries.extend(vector_summary_fields)
        summaries.append('}')

        return summaries

    def _get_vespa_type(self, marqo_type: FieldType) -> str:
        try:
            return self._MARQO_TO_VESPA_TYPE_MAP[marqo_type]
        except KeyError:
            raise InternalError(f'Unknown Marqo type: {marqo_type}')
