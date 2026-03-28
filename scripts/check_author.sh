#!/usr/bin/env bash
AUTHOR_NAME=$(git var GIT_AUTHOR_IDENT | sed 's/\(.*\) <.*/\1/')
AUTHOR_EMAIL=$(git var GIT_AUTHOR_IDENT | sed 's/.*<\(.*\)>.*/\1/')
if [ "$AUTHOR_NAME" != "Daniel Herman" ] || [ "$AUTHOR_EMAIL" != "daniel.herman@protonmail.com" ]; then
    echo "ERROR: Author must be Daniel Herman <daniel.herman@protonmail.com>"
    echo "Got: $AUTHOR_NAME <$AUTHOR_EMAIL>"
    exit 1
fi
