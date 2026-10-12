from datetime import date

from psr import backup


def test_backup_name_is_dated_so_names_sort_by_age():
    assert backup.name_for(date(2026, 10, 12)) == "bob-2026-10-12.sqlite.gz"


def test_keeps_the_newest_n_and_ignores_other_files():
    names = ["bob-2026-09-28.sqlite.gz", "notes.txt", "bob-2026-10-12.sqlite.gz",
             "bob-2026-10-05.sqlite.gz", "bob-2026-09-21.sqlite.gz"]
    assert backup.to_trash(names, keep=2) == ["bob-2026-09-21.sqlite.gz", "bob-2026-09-28.sqlite.gz"]


def test_nothing_to_trash_when_under_the_limit():
    assert backup.to_trash(["bob-2026-10-12.sqlite.gz"], keep=8) == []
