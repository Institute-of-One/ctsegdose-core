"""The anatomical screens, checked against the failures they exist to catch."""

from __future__ import annotations

from ctsegdose_core.checks import (
    check_left_right,
    check_masses,
    check_modulation,
    check_series,
    check_superior_inferior,
)


def organ(name, *, z, y=250.0, x=250.0, mass=100.0, weight=1.0):
    return {
        "organ": name,
        "centroid_zyx": [z, y, x],
        "mass_g": mass,
        "relative_weight": weight,
    }


def a_series(**overrides):
    series = {
        "series_instance_uid": "1.2.3",
        "vendor": "GE",
        "organs": [
            organ("liver", z=90.0, x=180.0, mass=1650.0, weight=0.99),
            organ("spleen", z=95.0, x=330.0, mass=160.0, weight=1.02),
            organ("kidney_left", z=70.0, x=310.0, mass=150.0, weight=1.01),
            organ("kidney_right", z=70.0, x=190.0, mass=148.0, weight=1.00),
            organ("adrenal_gland_left", z=88.0, x=300.0, mass=6.0, weight=1.03),
            organ("adrenal_gland_right", z=88.0, x=205.0, mass=5.0, weight=1.04),
            organ("pancreas", z=80.0, x=255.0, mass=130.0, weight=1.05),
            organ("urinary_bladder", z=15.0, x=255.0, mass=80.0, weight=1.09),
        ],
    }
    series.update(overrides)
    return series


def failures(checks):
    return [c.name for c in checks if c.is_failure]


def test_a_correct_segmentation_passes_every_screen():
    assert check_series(a_series())["passed"]


def test_a_mirrored_segmentation_is_caught_even_though_nothing_else_looks_wrong():
    """The failure this module exists for: volumes and masses stay perfectly plausible."""
    series = a_series()
    for o in series["organs"]:
        o["centroid_zyx"][2] = 512.0 - o["centroid_zyx"][2]
    report = check_series(series)
    assert not report["passed"]
    assert failures(check_left_right(series)) == [
        "left kidney is left of right kidney",
        "spleen is left of liver",
    ]


def test_a_head_foot_inversion_is_caught():
    series = a_series()
    for o in series["organs"]:
        o["centroid_zyx"][0] = 120.0 - o["centroid_zyx"][0]
    assert "liver is superior to bladder" in failures(check_superior_inferior(series))


def test_an_absent_organ_is_not_counted_as_a_failure():
    series = a_series(organs=[organ("liver", z=90.0, x=180.0, mass=1650.0)])
    assert all(c.passed for c in check_left_right(series))
    assert all(c.passed for c in check_superior_inferior(series))


def test_an_organ_mass_far_outside_the_reference_band_is_flagged():
    """A wrong voxel volume shows up here and nowhere else."""
    series = a_series()
    for o in series["organs"]:
        o["mass_g"] *= 3.0
    named = failures(check_masses(series))
    assert "liver mass plausible" in named
    assert "spleen mass plausible" in named


def test_the_band_is_wide_enough_for_real_anatomical_variation():
    series = a_series()
    for o in series["organs"]:
        o["mass_g"] *= 2.0   # a large but real adult
    assert all(c.passed for c in check_masses(series))


def test_an_organ_truncated_by_the_scan_is_not_judged_against_a_whole_organ_reference():
    """The Canon lung series: its inferior boundary cuts the upper abdomen, leaving a
    17 g pancreas. That is the field of view, not a segmentation defect."""
    series = a_series()
    pancreas = next(o for o in series["organs"] if o["organ"] == "pancreas")
    pancreas["mass_g"] = 17.0
    pancreas["truncated"] = True
    assert failures(check_masses(series)) == []
    report = check_series(series)
    assert report["truncated_organs"] == ["pancreas"]
    assert report["n_whole_organs"] == len(series["organs"]) - 1


def test_a_whole_organ_that_is_far_too_small_is_still_caught():
    series = a_series()
    next(o for o in series["organs"] if o["organ"] == "pancreas")["mass_g"] = 17.0
    assert "pancreas mass plausible" in failures(check_masses(series))


def test_only_solid_organs_are_size_checked():
    """The bladder is segmented with its contents; ICRP tabulates wall mass."""
    series = a_series()
    next(o for o in series["organs"] if o["organ"] == "urinary_bladder")["mass_g"] = 900.0
    assert all(c.passed for c in check_masses(series))


def test_flat_organ_weights_mean_the_modulation_is_not_being_applied():
    series = a_series()
    for o in series["organs"]:
        o["relative_weight"] = 1.0
    assert "organ weights vary" in failures(check_modulation(series))


def test_varying_weights_that_bracket_the_scan_mean_pass():
    assert all(c.passed for c in check_modulation(a_series()))


def test_organs_all_above_the_scan_mean_is_advisory_not_a_failure():
    """A chest-abdomen scan gives every abdominal organ more than the scan mean."""
    series = a_series()
    for i, o in enumerate(series["organs"]):
        o["relative_weight"] = 1.3 + 0.02 * i
    checks = check_modulation(series)
    assert failures(checks) == []
    bracket = next(c for c in checks if c.name == "weights bracket the scan mean")
    assert bracket.advisory and not bracket.passed


def test_checks_that_could_not_run_are_counted_not_hidden():
    """Half the organs missing must not read as a fully verified series."""
    report = check_series(a_series(organs=[organ("liver", z=90.0, x=180.0, mass=1650.0)]))
    assert report["passed"]
    assert report["n_not_applicable"] >= 5
