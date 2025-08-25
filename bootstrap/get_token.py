import os
import sys

os.environ.setdefault("DJANGO_SETTINGS_MODULE", "paperless.settings")

try:
    import django  # type: ignore
    django.setup()
except Exception:
    sys.exit(1)

try:
    from django.contrib.auth import get_user_model  # type: ignore
    from rest_framework.authtoken.models import Token  # type: ignore

    username = os.environ.get("PAPERLESS_ADMIN_USER", "admin")
    password = os.environ.get("PAPERLESS_ADMIN_PASSWORD", "")

    User = get_user_model()
    try:
        user = User.objects.get(username=username)
    except User.DoesNotExist:
        # Create superuser if it does not exist yet (first run)
        user = User.objects.create_superuser(username=username, email="", password=password)

    if not user.is_active:
        user.is_active = True
        user.save(update_fields=["is_active"])

    token, _ = Token.objects.get_or_create(user=user)
    print(token.key)
except Exception:
    sys.exit(1)
