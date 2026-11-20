"""Tests for --resume / --resume-uuid (continuing an interrupted TuningRun)."""

import argparse
from collections import Counter

import pytest

import opentuner
from opentuner import resultsdb
from opentuner.measurement import MeasurementInterface
from opentuner.resultsdb.models import DesiredResult, Result, TuningRun
from opentuner.search.differentialevolution import DifferentialEvolution
from opentuner.search.driver import SearchDriver
from opentuner.search.manipulator import (ConfigurationManipulator,
                                          IntegerParameter)
from opentuner.search.simplextechniques import SimplexTechnique
from opentuner.tuningrunmain import TuningRunMain


class Interrupted(Exception):
    """Stands in for the KeyboardInterrupt of a killed process."""


class TinyInterface(MeasurementInterface):
    def manipulator(self):
        manipulator = ConfigurationManipulator()
        manipulator.add_parameter(IntegerParameter('x', 0, 100))
        return manipulator

    def run(self, desired_result, input, limit):
        return Result(time=abs(desired_result.configuration.data['x'] - 42))

    def save_final_config(self, configuration):
        pass


class SeededTinyInterface(TinyInterface):
    """Requests the same configuration twice as seeds."""

    def seed_configurations(self):
        return [{'x': 50}, {'x': 50}]


class StopEarlySearchDriver(SearchDriver):
    """Behaves like a run whose process was killed after a few tests."""

    stop_after = 4

    def main(self):
        while not self.convergence_criteria():
            self.run_generation_techniques()
            self.run_generation_results()
            self.generation += 1
            if self.test_count >= self.stop_after:
                raise Interrupted()


def make_args(tmp_path, **overrides):
    parser = argparse.ArgumentParser(parents=opentuner.argparsers())
    args = parser.parse_args([])
    args.database = 'sqlite:///%s' % (tmp_path / 'opentuner.db')
    args.label = 'resume-test'
    args.test_limit = 30
    args.parallelism = 1
    args.seed_configuration = []
    for name, value in overrides.items():
        setattr(args, name, value)
    return args


def run_interrupted(tmp_path, interface_cls=TinyInterface, **overrides):
    args = make_args(tmp_path, **overrides)
    with pytest.raises(Interrupted):
        TuningRunMain(interface_cls(args), args,
                      search_driver=StopEarlySearchDriver).main()


def open_db(database):
    _engine, Session = resultsdb.connect(database)
    return Session()


def results_per_configuration(session):
    return Counter(result.configuration_id
                   for result in session.query(Result).all())


def test_resume_continues_the_interrupted_run(tmp_path):
    run_interrupted(tmp_path)
    args = make_args(tmp_path)

    session = open_db(args.database)
    try:
        run = session.query(TuningRun).one()
        assert run.state == 'ABORTED'
        interrupted_uuid = run.uuid
        recorded = session.query(DesiredResult).count()
        assert recorded >= StopEarlySearchDriver.stop_after
    finally:
        session.close()

    resume_args = make_args(tmp_path, resume=True)
    TuningRunMain(TinyInterface(resume_args), resume_args).main()

    session = open_db(resume_args.database)
    try:
        runs = session.query(TuningRun).all()
        # the same TuningRun was continued, not a second one created
        assert len(runs) == 1
        assert runs[0].uuid == interrupted_uuid
        assert runs[0].state == 'COMPLETE'
        # --test-limit counts every request ever made for the run, so the
        # resumed session stops at 30 in total rather than 4 + 30
        assert session.query(DesiredResult).count() == resume_args.test_limit
    finally:
        session.close()


def test_duplicate_request_reuses_the_existing_result(tmp_path):
    args = make_args(tmp_path, test_limit=6)
    TuningRunMain(SeededTinyInterface(args), args).main()

    session = open_db(args.database)
    try:
        seed_rows = (session.query(DesiredResult)
                     .filter_by(requestor='seed')
                     .order_by(DesiredResult.id).all())
        # the duplicate request is still recorded ...
        assert len(seed_rows) == 2
        # ... but it reuses the first request's Result instead of evaluating
        # the same configuration a second time
        assert seed_rows[0].result_id is not None
        assert seed_rows[0].result_id == seed_rows[1].result_id
        assert max(results_per_configuration(session).values()) == 1
    finally:
        session.close()


def test_resume_does_not_request_seed_configurations_again(tmp_path):
    run_interrupted(tmp_path, SeededTinyInterface)

    resume_args = make_args(tmp_path, resume=True)
    TuningRunMain(SeededTinyInterface(resume_args), resume_args).main()

    session = open_db(resume_args.database)
    try:
        # the two seed requests are the ones the interrupted process made;
        # the resumed session does not spend --test-limit budget on them again
        assert (session.query(DesiredResult)
                .filter_by(requestor='seed').count()) == 2
        assert session.query(DesiredResult).count() == resume_args.test_limit
        # and no configuration was evaluated twice across the two sessions
        assert max(results_per_configuration(session).values()) == 1
    finally:
        session.close()


def test_resume_requeues_requests_left_running(tmp_path):
    run_interrupted(tmp_path)
    args = make_args(tmp_path)

    session = open_db(args.database)
    try:
        # simulate a request the dead process had claimed but never reported
        session.query(DesiredResult).first().state = 'RUNNING'
        session.commit()
    finally:
        session.close()

    resume_args = make_args(tmp_path, resume=True)
    TuningRunMain(TinyInterface(resume_args), resume_args).main()

    session = open_db(resume_args.database)
    try:
        assert session.query(DesiredResult).filter_by(state='RUNNING').count() == 0
        # the re-queued request was measured, not skipped: the resumed session
        # still stops at the same total
        assert session.query(DesiredResult).count() == resume_args.test_limit
    finally:
        session.close()


def test_resume_replays_bandit_history(tmp_path):
    run_interrupted(tmp_path)

    args = make_args(tmp_path)
    session = open_db(args.database)
    try:
        measured = (session.query(DesiredResult)
                    .filter(DesiredResult.result_id.isnot(None)).count())
    finally:
        session.close()
    assert measured > 0

    resume_args = make_args(tmp_path, resume=True)
    trm = TuningRunMain(TinyInterface(resume_args), resume_args)
    try:
        trm.init()
        bandit = trm.search_driver.root_technique.bandit
        # every sub-technique starts with a zero count, so the totals are
        # entirely what the replay put back
        assert sum(bandit.use_counts.values()) == measured
        assert len(bandit.history) == measured
    finally:
        trm.session.close()


def test_resume_by_uuid(tmp_path):
    run_interrupted(tmp_path)
    args = make_args(tmp_path)

    session = open_db(args.database)
    try:
        interrupted_uuid = session.query(TuningRun).one().uuid
    finally:
        session.close()

    resume_args = make_args(tmp_path, resume_uuid=interrupted_uuid)
    TuningRunMain(TinyInterface(resume_args), resume_args).main()

    session = open_db(resume_args.database)
    try:
        assert session.query(TuningRun).one().uuid == interrupted_uuid
    finally:
        session.close()


def test_unknown_resume_uuid_raises(tmp_path):
    args = make_args(tmp_path, resume_uuid='deadbeef')
    with pytest.raises(ValueError):
        TuningRunMain(TinyInterface(args), args).main()


def test_without_resume_starts_a_new_run(tmp_path):
    run_interrupted(tmp_path)

    second_args = make_args(tmp_path)
    TuningRunMain(TinyInterface(second_args), second_args).main()

    session = open_db(second_args.database)
    try:
        assert session.query(TuningRun).count() == 2
    finally:
        session.close()


def test_resume_without_unfinished_run_starts_a_new_one(tmp_path):
    run_interrupted(tmp_path)

    # finish the interrupted run, so nothing is resumable any more
    resume_args = make_args(tmp_path, resume=True)
    TuningRunMain(TinyInterface(resume_args), resume_args).main()

    again_args = make_args(tmp_path, resume=True)
    TuningRunMain(TinyInterface(again_args), again_args).main()

    session = open_db(again_args.database)
    try:
        assert session.query(TuningRun).count() == 2
    finally:
        session.close()

def start_resumed_run(tmp_path):
    run_interrupted(tmp_path)
    args = make_args(tmp_path, resume=True)
    trm = TuningRunMain(TinyInterface(args), args)
    trm.init()
    return trm


def subtechnique(trm, cls):
    return next(technique
                for technique in trm.search_driver.root_technique.techniques
                if isinstance(technique, cls))


def test_top_results_are_distinct_and_best_first(tmp_path):
    trm = start_resumed_run(tmp_path)
    try:
        driver = trm.search_driver
        top = driver.top_results(30)
        assert top
        assert top[0].id == driver.best_result.id
        assert len(top) == len(set(result.configuration_id
                                  for result in top))
        assert all(result.state == 'OK' for result in top)
        assert len(driver.top_results(1)) == 1
        assert driver.top_results(0) == []
    finally:
        trm.session.close()


def test_top_results_skips_failed_and_duplicate_results(tmp_path):
    trm = start_resumed_run(tmp_path)
    try:
        driver = trm.search_driver
        best = driver.best_result

        duplicate = Result(configuration=best.configuration,
                           tuning_run=trm.tuning_run,
                           time=best.time)
        trm.session.add(duplicate)

        failed_config = driver.get_configuration({'x': 99})
        failed = Result(configuration=failed_config,
                        tuning_run=trm.tuning_run,
                        time=0.0, state='ERROR')
        trm.session.add(failed)
        trm.session.flush()

        top = driver.top_results(30)
        assert failed.id not in [result.id for result in top]
        assert len([result for result in top
                    if result.configuration_id == best.configuration.id]) == 1
        assert all(result.state == 'OK' for result in top)
    finally:
        trm.session.close()


def test_resume_warm_starts_differential_evolution(tmp_path):
    trm = start_resumed_run(tmp_path)
    try:
        driver = trm.search_driver
        top = driver.top_results(30)
        assert top

        de = subtechnique(trm, DifferentialEvolution)
        de.initial_population()

        seeded = [member for member in de.population if member.submitted]
        assert len(de.population) == de.population_size
        assert len(seeded) == len(top)
        assert [member.config.id for member in seeded] == [
            result.configuration_id for result in reversed(top)]
        random_members = de.population[:de.population_size - len(seeded)]
        assert not any(member.submitted for member in random_members)
        # best is last, so it is the last population member replaced
        assert de.population[-1].config.id == (
            driver.best_result.configuration.id)
    finally:
        trm.session.close()


def test_warm_start_fills_whole_population_when_enough_results(tmp_path):
    trm = start_resumed_run(tmp_path)
    try:
        top = trm.search_driver.top_results(30)
        assert top

        de = subtechnique(trm, DifferentialEvolution)
        de.population_size = len(top)
        de.initial_population()

        assert len(de.population) == len(top)
        assert all(member.submitted for member in de.population)
    finally:
        trm.session.close()


def test_resume_warm_starts_simplex_at_best(tmp_path):
    trm = start_resumed_run(tmp_path)
    try:
        driver = trm.search_driver
        simplex = subtechnique(trm, SimplexTechnique)
        assert simplex.seed_cfg is None

        simplex.desired_configuration()

        assert simplex.simplex_points
        assert simplex.simplex_points[0].id == (
            driver.best_result.configuration.id)
    finally:
        trm.session.close()


def test_explicit_simplex_seed_overrides_warm_start(tmp_path):
    trm = start_resumed_run(tmp_path)
    try:
        simplex = subtechnique(trm, SimplexTechnique)
        simplex.seed_cfg = {'x': 7}
        assert simplex.initial_simplex_seed() == {'x': 7}
    finally:
        trm.session.close()


def test_fresh_run_does_not_call_warm_start_query(tmp_path):
    # Warm start must be completely inert on a fresh run: no results exist,
    # so the techniques must not even query for them.
    args = make_args(tmp_path)
    trm = TuningRunMain(TinyInterface(args), args)
    trm.init()
    try:
        driver = trm.search_driver
        assert driver.best_result is None

        def fail(count):
            raise AssertionError('top_results called on a fresh run')

        driver.top_results = fail
        driver.run_generation_techniques()
        subtechnique(trm, DifferentialEvolution).initial_population()
    finally:
        trm.session.close()


def test_fresh_run_does_not_warm_start(tmp_path):
    args = make_args(tmp_path)
    trm = TuningRunMain(TinyInterface(args), args)
    trm.init()
    try:
        driver = trm.search_driver
        assert driver.best_result is None
        assert driver.top_results(10) == []

        de = subtechnique(trm, DifferentialEvolution)
        de.initial_population()
        assert len(de.population) == de.population_size
        assert not any(member.submitted for member in de.population)
    finally:
        trm.session.close()
