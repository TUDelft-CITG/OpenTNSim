# -*- coding: utf-8 -*-

"""Console script for opentnsim."""
# package(s) related to the command line interface
import sys

import click

import opentnsim.server


@click.group()
def cli(args=None):
    """OpenTNSim simulation."""
    click.echo("Replace this message by putting your code into " "opentnsim.cli.main")
    click.echo("See click documentation at http://click.pocoo.org/")
    return 0


@cli.command()
@click.option("--host", default="0.0.0.0")
@click.option("--port", default=5000, type=int)
@click.option("--debug/--no-debug", default=False)
def serve(host, port, debug, args=None):
    """Run a flask server with the backend code"""
    app = opentnsim.server.app
    app.run(host=host, debug=debug, port=port)


if __name__ == "__main__":
    sys.exit(cli())  # pragma: no cover
