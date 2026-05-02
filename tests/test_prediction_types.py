from plant_classifier.inference.types import PredictionLabel


def test_display_name_does_not_duplicate_genus_for_full_species_name() -> None:
    label = PredictionLabel(
        family="Fagaceae",
        genus="Quercus",
        species="Quercus rubra L.",
        score=0.9,
    )

    assert label.display_name == "Quercus rubra L."


def test_display_name_combines_genus_and_epithet() -> None:
    label = PredictionLabel(
        family="Fagaceae",
        genus="Quercus",
        species="rubra",
        score=0.9,
    )

    assert label.display_name == "Quercus rubra"
