import pprint
import time
from marqo.neural_search import enums, backend
from marqo.neural_search import neural_search
import unittest
import copy
from marqo.errors import InvalidArgError, IndexNotFoundError
from marqo.client import Client
from tests.marqo_test import MarqoTestCase


class TestlexicalSearch(MarqoTestCase):

    def setUp(self) -> None:
        self.generic_header = {"Content-type": "application/json"}
        self.client = Client(**self.client_settings)
        self.index_name_1 = "my-test-index-1"
        self.config = copy.deepcopy(self.client.config)
        try:
            self.client.delete_index(self.index_name_1)
        except IndexNotFoundError as s:
            pass
    
    @staticmethod
    def strip_marqo_fields(doc, strip_id=False):
        """Strips Marqo fields from a returned doc to get the original doc"""
        copied = copy.deepcopy(doc)

        strip_fields = ["_highlights", "_score"]
        if strip_id:
            strip_fields += ["_id"]

        for to_strip in strip_fields:
            try:
                del copied[to_strip]
            except KeyError:
                pass
        return copied

    
    def test_lexical_search_empty_text(self):
        neural_search.add_documents(
            config=self.config, index_name=self.index_name_1,
            docs=[{"some doc 1": "some field 2", "some doc 2": "some other thing"}], auto_refresh=True)
        res = neural_search._lexical_search(config=self.config, index_name=self.index_name_1, text="")
        assert len(res["hits"]) == 0
        assert res["hits"] == []

    def test_lexical_search_bad_text_type(self):
        neural_search.add_documents(
            config=self.config, index_name=self.index_name_1,
            docs=[{"some doc 1": "some field 2", "some doc 2": "some other thing"}], auto_refresh=True)
        bad_args = [None, 1234, 1.0]
        for a in bad_args:
            try:
                res = neural_search._lexical_search(config=self.config, index_name=self.index_name_1, text=a)
                raise AssertionError
            except InvalidArgError as e:
                assert "type str" in str(e)

    def test_lexical_search_no_index(self):
        try:
            res = neural_search._lexical_search(config=self.config, index_name="non existent index", text="abcdefg")
        except IndexNotFoundError as s:
            pass

    def test_lexical_search_multiple(self):
        d0 = {
            "some doc 1": "some FIELD 2", "_id": "alpha alpha",
            "the big field": "very unlikely theory. marqo is pretty awesom, in the field"
        }
        d1 = {"title": "Marqo", "some doc 2": "some other thing", "_id": "abcdef"}
        neural_search.add_documents(
            config=self.config, index_name=self.index_name_1, auto_refresh=True,
            docs=[d1, {"some doc 1": "some 2", "field abc": "robodog is not a cat", "_id": "unusual id"},
                  d0])
        res = neural_search._lexical_search(config=self.config, index_name=self.index_name_1, text="marqo field",
                                            return_doc_ids=True)
        assert len(res["hits"]) == 2
        assert res["hits"][0]["_id"] == "alpha alpha"
        assert res["hits"][1]["_id"] == "abcdef"
        assert self.strip_marqo_fields(res["hits"][0]) == d0
        assert self.strip_marqo_fields(res["hits"][1]) == d1

    def test_lexical_search_single_searchable_attribs(self):
        d0 = {
            "some doc 1": "some FIELD 2", "_id": "alpha alpha",
            "the big field": "very unlikely theory. marqo is pretty awesom, in the field"
        }
        d1 = {"title": "Marqo", "some doc 2": "some other thing", "_id": "abcdef"}
        d2 = {"some doc 1": "some 2 jnkerkbj", "field abc": "robodog is not a cat", "_id": "Jupyter_12"}
        d3 = {"TITITLE": "Tony from the way", "field lambda": "some prop field called marqo",
              "_id": "122"}
        d4 = {"Lucy": "Travis", "field lambda": "there is a whole bunch of text here. "
                                                "Just a slight mention of a field", "_id": "123"}
        neural_search.add_documents(
            config=self.config, index_name=self.index_name_1, auto_refresh=True,
            docs=[d0, d4, d1 ])
        neural_search.add_documents(
            config=self.config, index_name=self.index_name_1, auto_refresh=True,
            docs=[d3, d2])
        res = neural_search._lexical_search(
            config=self.config, index_name=self.index_name_1, text="marqo field",
            return_doc_ids=True, searchable_attributes=["field lambda"], result_count=3)
        assert len(res["hits"]) == 2
        assert res["hits"][0]["_id"] == "122"
        assert res["hits"][1]["_id"] == "123"
        assert self.strip_marqo_fields(res["hits"][0], strip_id=False) == d3
        assert self.strip_marqo_fields(res["hits"][1]) == d4

    def test_lexical_search_multiple_searchable_attribs(self):
        d0 = {
            "some doc 1": "some FIELD 2", "_id": "alpha alpha",
            "the big field": "very unlikely theory. marqo is pretty awesom, in the field"
        }
        d1 = {"FIELD omega": "Marqo which has many fields ", "field lambda": "field", "_id": "abcdef"}
        d2 = {"some doc 1": "some 2 jnkerkbj", "field abc": "robodog is not a cat", "_id": "Jupyter_12"}
        d3 = {"TITITLE": "Tony from the way", "_id": "122",
              "field lambda": "some prop called marqo. This actually has a lot more content than you thought." }
        d4 = {"Lucy": "Travis", "field lambda": "there is a whole bunch of text here. "
                                                "Just a slight mention of a field", "_id": "123"}
        neural_search.add_documents(
            config=self.config, index_name=self.index_name_1, auto_refresh=True,
            docs=[d0, d4, d1])
        neural_search.add_documents(
            config=self.config, index_name=self.index_name_1, auto_refresh=True,
            docs=[d3, d2])
        res = neural_search._lexical_search(
            config=self.config, index_name=self.index_name_1, text="Marqo field",
            return_doc_ids=True, searchable_attributes=["field lambda", "FIELD omega"])
        assert len(res["hits"]) == 3
        assert self.strip_marqo_fields(res["hits"][0]) == d1
        assert self.strip_marqo_fields(res["hits"][1]) == d3
        assert self.strip_marqo_fields(res["hits"][2]) == d4

    def test_lexical_search_multiple_searchable_attribs_no_returned_ids(self):
        d0 = {"some doc 1": "some FIELD 2",
            "the big field": "very unlikely theory. marqo marqo field is pretty awesom, in the marqo field"
        }
        d1 = {
            # SHOULD APPEAR FIRST!
            "FIELD omega": "sentence with the word marqo field awks ",
            "field lambda":  "sentence with the word marqo field awks ",}
        d2 = {"some doc 1": "some 2 jnkerkbj", "field abc": "robodog is not a cat field field field field field"}
        d3 = {"TITITLE": "Tony from the way", # SHOULD APPEAR SECOND
              "field lambda": "sentence with the word marqo field " }
        d4 = { # SHOULD APPEAR 3rd (LAST)
            "Lucy": "Travis", "field lambda": "sentence with the word field" }
        neural_search.add_documents(
            config=self.config, index_name=self.index_name_1, auto_refresh=True,
            docs=[d0, d4, d1])
        neural_search.add_documents(
            config=self.config, index_name=self.index_name_1, auto_refresh=True,
            docs=[d3, d2])
        time.sleep(1)
        res = neural_search._lexical_search(
            config=self.config, index_name=self.index_name_1, text="Marqo field awks",
            return_doc_ids=False, searchable_attributes=["field lambda", "FIELD omega"],
            result_count=3)
        assert len(res["hits"]) == 3
        assert self.strip_marqo_fields(res["hits"][0]) == d1
        assert self.strip_marqo_fields(res["hits"][1]) == d3
        assert self.strip_marqo_fields(res["hits"][2]) == d4

    def test_lexical_search_non_existent_searchable_attrib(self):
        d0 = {
            "some doc 1": "some FIELD 2",
            "the big field": "very unlikely theory. marqo is pretty awesom, in the field"
        }
        neural_search.add_documents(
            config=self.config, index_name=self.index_name_1, auto_refresh=True,
            docs=[d0])
        try:
            res = neural_search._lexical_search(
                config=self.config, index_name=self.index_name_1, text="Marqo field",
                return_doc_ids=False, searchable_attributes=["never existed field hehehehe"],
                raise_for_searchable_attributes=True
            )
            raise AssertionError
        except InvalidArgError as s:
            assert "unknown searchable_attributes" in str(s)

    def test_lexical_search_result_count(self):
        d0 = {
            "some doc 1": "some FIELD 2",
            "the big field": "very unlikely theory. marqo is pretty awesom, in the field"
        }
        d1 = {"FIELD omega": "Marqo which has many fields ", "field lambda": "field",}
        d2 = {"some doc 1": "some 2 jnkerkbj", "field abc": "robodog is not a cat"}
        d3 = {"TITITLE": "Tony from the way",
              "field lambda": "some prop called marqo. This actually has a lot more content than you thought." }
        d4 = {"Lucy": "Travis", "field lambda": "there is a whole bunch of text here. Some other text. "
                                                "Trying to reduce often the keywords appear here. SMH "
                                                "Another bunch of words that may mean something. "
                                                "Just a slight mention of a field"}
        d5 = {"some completely irrelevant": "document hehehe"}
        neural_search.add_documents(
            config=self.config, index_name=self.index_name_1, auto_refresh=True,
            docs=[d0, d4, d1, d3, d2])
        r1 = neural_search._lexical_search(
            config=self.config, index_name=self.index_name_1, text="Marqo field",
            return_doc_ids=False, result_count=2
        )
        assert len(r1["hits"]) == 2
        r2 = neural_search._lexical_search(
            config=self.config, index_name=self.index_name_1, text="Marqo field",
            return_doc_ids=False, result_count=1000
        )
        assert len(r2["hits"]) == 4
        r3 = neural_search._lexical_search(
            config=self.config, index_name=self.index_name_1, text="Marqo field",
            return_doc_ids=False, result_count=0
        )
        assert len(r3["hits"]) == 0

    def test_search_lexical_param(self):
        """neural_search.search(search_method='lexical') should behave identically to
            lexical_search() for a given set of args
        """
        d0 = {
            "some doc 1": "some FIELD 2",
            "the big field": "very unlikely theory. marqo is pretty awesom, in the field"
        }
        d1 = {"FIELD omega": "Marqo which has many fields ", "field lambda": "field",}
        d2 = {"some doc 1": "some 2 jnkerkbj", "field abc": "robodog is not a cat"}
        d3 = {"TITITLE": "Tony from the way",
              "field lambda": "some prop called marqo. This actually has a lot more content than you thought." }
        d4 = {"Lucy": "Travis", "field lambda": "there is a whole bunch of text here. "
                                                "Just a slight mention of a field"}
        d5 = {"some completely irrelevant": "document hehehe"}
        neural_search.add_documents(
            config=self.config, index_name=self.index_name_1, auto_refresh=True,
            docs=[d0, d4, d1, d3, d2])
        res_lexical_search = neural_search._lexical_search(
            config=self.config, index_name=self.index_name_1, text="Marqo field",
            return_doc_ids=False, searchable_attributes=["field lambda", "FIELD omega"])
        res_search_entry_point = neural_search.search(
            config=self.config, index_name=self.index_name_1, text="Marqo field",
            return_doc_ids=False, searchable_attributes=["field lambda", "FIELD omega"],
            search_method=enums.SearchMethod.LEXICAL)
        res_search_entry_point_no_processing_time = res_search_entry_point.copy()
        del res_search_entry_point_no_processing_time ['processingTimeMs']
        del res_search_entry_point_no_processing_time ['query']
        del res_search_entry_point_no_processing_time ['limit']
        assert len(res_lexical_search['hits']) > 0
        assert res_search_entry_point_no_processing_time == res_lexical_search


    def test_lexical_search_overwriting_doc(self):
        """can we overwrite doc and do a search on the latest doc?"""
        a_consistent_id = "my id is awesome"
        d0 = {
            "_id": a_consistent_id,
            "some doc 1": "some  2",
            "the big field": "very unlikely theory. is pretty awesom, in the",
            "boring field": "4 grey boring walls. "
        }
        d1 = {
            "_id": a_consistent_id,
            "some doc 1": "some FIELD 2",
            "the big field": "just your average doc...",
            "Cool field": "Marqo is the best!"
        }
        neural_search.add_documents(
            config=self.config, index_name=self.index_name_1, auto_refresh=True,
            docs=[d0])
        assert [] == neural_search._lexical_search(
            config=self.config, index_name=self.index_name_1, text="Marqo field",
            return_doc_ids=False)["hits"]
        grey_query = neural_search._lexical_search(
            config=self.config, index_name=self.index_name_1, text="4 grey boring walls",
            return_doc_ids=True)
        assert len (grey_query["hits"]) == 1
        assert grey_query["hits"][0]["_id"] == a_consistent_id
        # update doc so it does indeed get returned
        neural_search.add_documents(
            config=self.config, index_name=self.index_name_1, auto_refresh=True,
            docs=[d1])
        cool_query = neural_search._lexical_search(
            config=self.config, index_name=self.index_name_1, text="Marqo field",
            return_doc_ids=True)
        assert a_consistent_id == cool_query["hits"][0]["_id"]
        assert len(cool_query["hits"]) == 1
        assert [] == neural_search._lexical_search(
            config=self.config, index_name=self.index_name_1, text="4 grey boring walls",
            return_doc_ids=False)["hits"]
