"""Límite de mensajes por usuario y por día.

Los contadores viven en memoria del servidor (compartidos entre todas las
sesiones gracias a ``st.cache_resource``) y se reinician automáticamente
cada día a la medianoche de Colombia. Si la app se reinicia o se "duerme"
en Streamlit Cloud, los contadores también vuelven a cero.
"""

from __future__ import annotations

import threading
from datetime import datetime
from zoneinfo import ZoneInfo

ZONA_HORARIA = ZoneInfo("America/Bogota")


def _hoy() -> str:
    return datetime.now(ZONA_HORARIA).date().isoformat()


class LimitadorMensajes:
    """Cuenta cuántos mensajes envía cada usuario por día.

    Además del límite por usuario hay un tope global diario: es la red de
    seguridad que protege tu saldo de OpenAI aunque alguien logre saltarse
    el límite individual (por ejemplo, cambiando de red o de IP).
    """

    def __init__(self, max_por_usuario: int, max_global: int) -> None:
        self.max_por_usuario = max_por_usuario
        self.max_global = max_global
        self._lock = threading.Lock()
        self._dia = _hoy()
        self._por_usuario: dict[str, int] = {}
        self._total = 0

    def _reiniciar_si_cambio_el_dia(self) -> None:
        dia_actual = _hoy()
        if dia_actual != self._dia:
            self._dia = dia_actual
            self._por_usuario.clear()
            self._total = 0

    def restantes(self, usuario: str) -> int:
        with self._lock:
            self._reiniciar_si_cambio_el_dia()
            usados = self._por_usuario.get(usuario, 0)
            return max(0, self.max_por_usuario - usados)

    def intentar_consumir(self, usuario: str) -> str | None:
        """Descuenta un mensaje. Devuelve ``None`` si se permitió o el
        motivo del rechazo: ``"usuario"`` o ``"global"``."""
        with self._lock:
            self._reiniciar_si_cambio_el_dia()
            if self._total >= self.max_global:
                return "global"
            if self._por_usuario.get(usuario, 0) >= self.max_por_usuario:
                return "usuario"
            self._por_usuario[usuario] = self._por_usuario.get(usuario, 0) + 1
            self._total += 1
            return None

    def devolver(self, usuario: str) -> None:
        """Reintegra un mensaje (se usa cuando OpenAI falla y no hubo respuesta)."""
        with self._lock:
            if self._por_usuario.get(usuario, 0) > 0:
                self._por_usuario[usuario] -= 1
                self._total = max(0, self._total - 1)
