"""
Tests for the knowledge base built by tools/build_knowledge_base.py.

The file is checked for well-formedness (it re-reads to the same graph and
is written the same way twice), OWL DL hygiene (no IRI with two kinds, no
dangling references, property assertions that respect domain and range),
and the inferences it is designed for, run with the pure-Python OWL RL
reasoner (owlrl). The two classes defined by a numeric range (deep /
shallow architecture) need a reasoner with datatype facets, such as HermiT
in Protégé, so they are not checked here.
"""

from typing import TYPE_CHECKING

import pytest

rdflib = pytest.importorskip("rdflib")
owlrl = pytest.importorskip("owlrl")

from rdflib import OWL, RDF, RDFS, Graph, URIRef
from rdflib.compare import isomorphic
from rdflib.term import Node
from tools.build_knowledge_base import NNFS, build_graph, write_rdfxml

if TYPE_CHECKING:
    # Only for the tmp_path annotation; the tests themselves never import pathlib.
    from pathlib import Path


@pytest.fixture(scope="module")
def graph() -> Graph:
    """
    The knowledge base as built from the repository, once per module.

    Returns
    -------
    rdflib.Graph
        The asserted ontology.

    Notes
    -----
    Processing:
    1. Call build_graph() once and share it between the tests.
    """
    return build_graph()


@pytest.fixture(scope="module")
def inferred(graph: Graph) -> Graph:
    """
    The knowledge base after OWL RL reasoning.

    Parameters
    ----------
    graph : rdflib.Graph
        The asserted ontology.

    Returns
    -------
    rdflib.Graph
        A copy with the OWL RL deductive closure added.

    Notes
    -----
    Processing:
    1. Copy the graph and expand it with owlrl's OWL RL semantics.
    """
    copy = Graph()
    for triple in graph:
        copy.add(triple)
    owlrl.DeductiveClosure(owlrl.OWLRL_Semantics).expand(copy)
    return copy


def test_written_file_reads_back_to_the_same_graph(graph: Graph, tmp_path: "Path") -> None:
    """
    write_rdfxml produces RDF/XML that parses to an isomorphic graph, twice over.

    Parameters
    ----------
    graph : rdflib.Graph
        The asserted ontology (module-scoped fixture).
    tmp_path : pathlib.Path
        Empty temporary directory provided by pytest.

    Notes
    -----
    Round-tripping through rdflib's parser must give a graph isomorphic to
    the one built (blank nodes matched structurally), and writing twice
    must give byte-identical files, so regenerating the knowledge base
    never produces spurious diffs.
    """
    first, second = tmp_path / "a.owl", tmp_path / "b.owl"
    write_rdfxml(graph, first)
    write_rdfxml(build_graph(), second)
    parsed = Graph().parse(str(first), format="xml")
    assert len(parsed) == len(graph)
    assert isomorphic(parsed, graph)
    assert first.read_bytes() == second.read_bytes()


def test_no_entity_has_two_kinds_and_no_reference_is_dangling(graph: Graph) -> None:
    """
    Every nnfs IRI is declared exactly once as a class, property or individual.

    Parameters
    ----------
    graph : rdflib.Graph
        The asserted ontology (module-scoped fixture).

    Notes
    -----
    OWL DL forbids using one IRI as, say, both an individual and a
    property (the package "tests" and a property "tests" would clash).
    Every nnfs IRI that appears anywhere in the graph must also be
    declared, so no assertion points at a misspelled name.
    """
    kinds = [
        OWL.Class,
        OWL.ObjectProperty,
        OWL.DatatypeProperty,
        OWL.AnnotationProperty,
        OWL.NamedIndividual,
    ]
    declared: dict[URIRef, URIRef] = {}
    for kind in kinds:
        for entity in graph.subjects(RDF.type, kind):
            if isinstance(entity, URIRef):
                assert entity not in declared, f"{entity} is both {declared[entity]} and {kind}"
                declared[entity] = kind
    used = {
        term
        for triple in graph
        for term in triple
        if isinstance(term, URIRef) and str(term).startswith(str(NNFS))
    }
    assert used - set(declared) == set()
    # rdflib names generated blank nodes b0, b1, ...; one leaking into the nnfs namespace
    # would mean a missing name was filled in with a blank-node id.
    assert not any(str(e)[len(str(NNFS)) :].startswith("b0") for e in declared)


def test_assertions_respect_domains_and_ranges(graph: Graph) -> None:
    """
    Every property assertion between individuals matches the property's domain and range.

    Parameters
    ----------
    graph : rdflib.Graph
        The asserted ontology (module-scoped fixture).

    Notes
    -----
    For each object property with a declared domain or range, the subject
    (object) of every assertion must be asserted to be in that class or
    one of its subclasses. The same check is applied to data properties'
    domains. A violation would make the ontology inconsistent in Protégé.
    """
    subclasses: dict[Node, set[Node]] = {}
    for sub, sup in graph.subject_objects(RDFS.subClassOf):
        if isinstance(sub, URIRef) and isinstance(sup, URIRef):
            subclasses.setdefault(sup, set()).add(sub)

    def descendants(cls: Node) -> set[Node]:
        """
        A class together with all its direct and indirect subclasses.

        Parameters
        ----------
        cls : rdflib.term.Node
            The class to start from, e.g. a property's domain.

        Returns
        -------
        set of rdflib.term.Node
            cls and every named class below it in the rdfs:subClassOf tree.

        Notes
        -----
        Processing:
        1. Start with cls as the only found class and the only frontier entry.
        2. Pop a class, add its not-yet-found direct subclasses to both sets.
        3. Stop when the frontier is empty.

        A plain transitive closure, written out so the test doesn't depend
        on the reasoner whose results the other tests check.
        """
        found, frontier = {cls}, [cls]
        while frontier:
            new = subclasses.get(frontier.pop(), set()) - found
            found |= new
            frontier.extend(new)
        return found

    def types(individual: Node) -> set[Node]:
        """
        The classes an individual is asserted (not inferred) to belong to.

        Parameters
        ----------
        individual : rdflib.term.Node
            Subject or object of a property assertion.

        Returns
        -------
        set of rdflib.term.Node
            Every object of an rdf:type triple about the individual.
        """
        return set(graph.objects(individual, RDF.type))

    violations = []
    for prop in set(graph.subjects(RDF.type, OWL.ObjectProperty)) | set(
        graph.subjects(RDF.type, OWL.DatatypeProperty)
    ):
        domain = graph.value(prop, RDFS.domain)
        # Only object properties' ranges are classes; a data property's range is a
        # datatype (e.g. xsd:integer), which no individual is typed with.
        range_ = (
            graph.value(prop, RDFS.range)
            if (prop, RDF.type, OWL.ObjectProperty) in graph
            else None
        )
        for subject, obj in graph.subject_objects(prop):
            if domain is not None and not types(subject) & descendants(domain):
                violations.append((prop, subject, "domain"))
            if range_ is not None and not types(obj) & descendants(range_):
                violations.append((prop, obj, "range"))
    assert violations == []


def test_reasoner_infers_the_pipeline_order(inferred: Graph) -> None:
    """
    Data flow through artifacts and imports becomes feeds and upstreamOf edges.

    Parameters
    ----------
    inferred : rdflib.Graph
        The ontology after OWL RL reasoning (module-scoped fixture).

    Notes
    -----
    train.py writes the results JSON that analysis.py reads, so the chain
    writes ∘ readBy must give "train feeds analysis". Through reads,
    imports, writes and transitivity, the classification config must be
    upstream of the benchmark report, and the benchmark report must not be
    upstream of the config.
    """
    train = NNFS["nn_numpy.modeling.train"]
    assert (train, NNFS.feeds, NNFS["nn_numpy.analysis"]) in inferred
    assert (NNFS.ClassificationConfig, NNFS.upstreamOf, NNFS.BenchmarkReport) in inferred
    assert (NNFS["nn_numpy.nn.layers"], NNFS.upstreamOf, NNFS.MainResultsJSON) in inferred
    assert (NNFS.BenchmarkReport, NNFS.upstreamOf, NNFS.ClassificationConfig) not in inferred


def test_reasoner_infers_stage_membership_and_roles(inferred: Graph) -> None:
    """
    Modules inherit their package's stage, and module roles are classified.

    Parameters
    ----------
    inferred : rdflib.Graph
        The ontology after OWL RL reasoning (module-scoped fixture).

    Notes
    -----
    layers.py asserts only partOf nn_numpy.nn; the chain
    partOf ∘ belongsToStage must place it in the network-library stage.
    train.py reads nothing itself but writes results and is run by make
    train, so it is a Producer and an EntryPoint but not a PipelineStep;
    benchmarks/report.py reads and writes, so it is a PipelineStep. The
    Keras wrapper needs TensorFlow, an optional library.
    """
    layers = NNFS["nn_numpy.nn.layers"]
    assert (layers, NNFS.belongsToStage, NNFS.NetworkLibraryStage) in inferred
    assert (layers, RDF.type, NNFS.NetworkLibraryStageComponent) in inferred
    assert (layers, RDF.type, NNFS.TestedModule) in inferred
    train = NNFS["nn_numpy.modeling.train"]
    assert (train, RDF.type, NNFS.Producer) in inferred
    assert (train, RDF.type, NNFS.EntryPoint) in inferred
    assert (train, RDF.type, NNFS.PipelineStep) not in inferred
    report = NNFS["nn_numpy.benchmarks.report"]
    assert (report, RDF.type, NNFS.PipelineStep) in inferred
    assert (
        NNFS["nn_numpy.benchmarks.keras_models"],
        RDF.type,
        NNFS.OptionalModule,
    ) in inferred
    assert (NNFS["nn_numpy.nn.network"], RDF.type, NNFS.OptionalModule) not in inferred
    assert (NNFS.MainResultsJSON, RDF.type, NNFS.StampedArtifact) in inferred
    assert (NNFS.ClassificationConfig, RDF.type, NNFS.StampedArtifact) not in inferred


def test_reasoner_infers_architecture_families(inferred: Graph) -> None:
    """
    Architectures are classified by the techniques they use and the base they derive from.

    Parameters
    ----------
    inferred : rdflib.Graph
        The ontology after OWL RL reasoning (module-scoped fixture).

    Notes
    -----
    A2-bn-dropout uses batch normalization and dropout, so it must be both
    a NormalizedArchitecture and a RegularizedArchitecture; A2-clip a
    ClippedArchitecture; A2-bias a Variant (of A2) but A1 not. The Keras
    A1 mirrors an architecture and so is a FrameworkReimplementation; the
    scikit-learn MLP is not.
    """
    assert (NNFS.A2_bn_dropout, RDF.type, NNFS.NormalizedArchitecture) in inferred
    assert (NNFS.A2_bn_dropout, RDF.type, NNFS.RegularizedArchitecture) in inferred
    assert (NNFS.A2_clip, RDF.type, NNFS.ClippedArchitecture) in inferred
    assert (NNFS.A2_bias, RDF.type, NNFS.Variant) in inferred
    assert (NNFS.A2_bias, NNFS.variantOf, NNFS.A2) in inferred
    assert (NNFS.A1, RDF.type, NNFS.Variant) not in inferred
    assert (NNFS.tensorflow_A1, RDF.type, NNFS.FrameworkReimplementation) in inferred
    assert (
        NNFS.sklearn_classification_mlp,
        RDF.type,
        NNFS.FrameworkReimplementation,
    ) not in inferred


def test_property_characteristics_hold_and_respect_owl_dl(graph: Graph, inferred: Graph) -> None:
    """
    Functional and inverse-functional properties are used consistently and only on simple properties.

    Parameters
    ----------
    graph : rdflib.Graph
        The asserted ontology (module-scoped fixture).
    inferred : rdflib.Graph
        The ontology after OWL RL reasoning (module-scoped fixture).

    Notes
    -----
    OWL 2 DL forbids functional / inverse-functional characteristics on
    non-simple properties: transitive ones, ones defined by a property
    chain, and ones with such a sub-property. Every subject of a functional
    property, and every object of an inverse-functional one, must have at
    most one partner, in the asserted and in the inferred graph; otherwise
    the reasoner would merge distinct individuals (owl:sameAs). Every
    property with an inverse must be transitive exactly when its inverse
    is.
    """
    non_simple = set(graph.subjects(RDF.type, OWL.TransitiveProperty)) | set(
        graph.subjects(OWL.propertyChainAxiom, None)
    )
    # Non-simplicity spreads to super-properties and inverses; iterate to a fixed point
    # because one pass can expose new parents.
    changed = True
    while changed:
        before = len(non_simple)
        non_simple |= {q for p, q in graph.subject_objects(RDFS.subPropertyOf) if p in non_simple}
        non_simple |= {q for p, q in graph.subject_objects(OWL.inverseOf) if p in non_simple}
        non_simple |= {p for p, q in graph.subject_objects(OWL.inverseOf) if q in non_simple}
        changed = len(non_simple) != before

    functional = set(graph.subjects(RDF.type, OWL.FunctionalProperty))
    inverse_functional = set(graph.subjects(RDF.type, OWL.InverseFunctionalProperty))
    object_properties = set(graph.subjects(RDF.type, OWL.ObjectProperty))
    assert (functional | inverse_functional) & object_properties & non_simple == set()
    # Lower bounds, not exact counts (12 and 12 when written): the check below must not pass
    # vacuously, yet adding a property must not break the test.
    assert len(functional & object_properties) >= 5 and len(inverse_functional) >= 5

    for g in (graph, inferred):
        for prop in functional & object_properties:
            for subject in set(g.subjects(prop, None)):
                assert len(set(g.objects(subject, prop))) == 1, (prop, subject)
        for prop in inverse_functional:
            for obj in set(g.objects(None, prop)):
                assert len(set(g.subjects(prop, obj))) == 1, (prop, obj)
    named = [
        (a, b)
        for a, b in inferred.subject_objects(OWL.sameAs)
        if a != b and str(a).startswith(str(NNFS)) and str(b).startswith(str(NNFS))
    ]
    assert named == []

    transitive = set(graph.subjects(RDF.type, OWL.TransitiveProperty))
    for p, q in graph.subject_objects(OWL.inverseOf):
        assert (p in transitive) == (q in transitive), (p, q)


def test_symmetric_asymmetric_and_irreflexive_properties_hold(
    graph: Graph, inferred: Graph
) -> None:
    """
    Symmetric, asymmetric and irreflexive properties are valid OWL 2 DL and true of the data.

    Parameters
    ----------
    graph : rdflib.Graph
        The asserted ontology (module-scoped fixture).
    inferred : rdflib.Graph
        The ontology after OWL RL reasoning (module-scoped fixture).

    Notes
    -----
    Asymmetric and irreflexive characteristics are only allowed on simple
    properties (not transitive, not chain-defined). No asymmetric property
    may hold in both directions between two individuals and no irreflexive
    one may relate an individual to itself, in the asserted or the inferred
    graph. The symmetric frameworkCounterpartOf, asserted once per pair
    (PyTorch -> Keras), must be inferred in the other direction too.
    """
    non_simple = set(graph.subjects(RDF.type, OWL.TransitiveProperty)) | set(
        graph.subjects(OWL.propertyChainAxiom, None)
    )
    asymmetric = set(graph.subjects(RDF.type, OWL.AsymmetricProperty))
    irreflexive = set(graph.subjects(RDF.type, OWL.IrreflexiveProperty))
    assert (asymmetric | irreflexive) & non_simple == set()
    # A lower bound (48 when written), for the same reason as the functional-property floor.
    assert len(asymmetric) >= 40

    for g in (graph, inferred):
        for prop in asymmetric:
            assert [(a, b) for a, b in g.subject_objects(prop) if (b, prop, a) in g] == [], prop
        for prop in asymmetric | irreflexive:
            assert [a for a, b in g.subject_objects(prop) if a == b] == [], prop

    counterpart = NNFS.frameworkCounterpartOf
    assert (counterpart, RDF.type, OWL.SymmetricProperty) in graph
    assert (NNFS.pytorch_A2_bn, counterpart, NNFS.tensorflow_A2_bn) in graph
    assert (NNFS.tensorflow_A2_bn, counterpart, NNFS.pytorch_A2_bn) in inferred
