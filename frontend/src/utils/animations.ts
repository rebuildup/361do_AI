/**
 * Animation utilities that respect user preferences and provide smooth interactions
 */

import { prefersReducedMotion, getAnimationDuration } from './accessibility';

/**
 * Animation configuration
 */
export const ANIMATION_CONFIG = {
  duration: {
    fast: 150,
    normal: 200,
    slow: 300,
  },
  easing: {
    easeOut: 'cubic-bezier(0.0, 0.0, 0.2, 1)',
    easeIn: 'cubic-bezier(0.4, 0.0, 1, 1)',
    easeInOut: 'cubic-bezier(0.4, 0.0, 0.2, 1)',
  },
};

/**
 * Creates CSS transition string with user preference consideration
 */
export const createTransition = (
  property: string,
  duration: keyof typeof ANIMATION_CONFIG.duration = 'normal',
  easing: keyof typeof ANIMATION_CONFIG.easing = 'easeOut'
): string => {
  const actualDuration = getAnimationDuration(
    ANIMATION_CONFIG.duration[duration]
  );
  return `${property} ${actualDuration}ms ${ANIMATION_CONFIG.easing[easing]}`;
};

/**
 * Animation classes that respect reduced motion preferences
 */
export const getAnimationClasses = () => {
  const reducedMotion = prefersReducedMotion();

  return {
    fadeIn: reducedMotion ? 'opacity-100' : 'animate-fade-in',
    fadeOut: reducedMotion ? 'opacity-0' : 'animate-fade-out',
    slideInRight: reducedMotion
      ? 'translate-x-0 opacity-100'
      : 'animate-slide-in-right',
    slideInLeft: reducedMotion
      ? 'translate-x-0 opacity-100'
      : 'animate-slide-in-left',
    slideOutRight: reducedMotion
      ? 'translate-x-full opacity-0'
      : 'animate-slide-out-right',
    slideOutLeft: reducedMotion
      ? '-translate-x-full opacity-0'
      : 'animate-slide-out-left',
    scaleIn: reducedMotion ? 'scale-100 opacity-100' : 'animate-scale-in',
    scaleOut: reducedMotion ? 'scale-95 opacity-0' : 'animate-scale-out',
    bounceSubtle: reducedMotion ? '' : 'animate-bounce-subtle',
    pulseSubtle: reducedMotion ? '' : 'animate-pulse-subtle',
    spin: reducedMotion ? '' : 'animate-spin',
  };
};

/**
 * Smooth scroll utility
 */
export const smoothScrollTo = (
  element: HTMLElement,
  options: ScrollIntoViewOptions = {}
) => {
  const defaultOptions: ScrollIntoViewOptions = {
    behavior: prefersReducedMotion() ? 'auto' : 'smooth',
    block: 'nearest',
    inline: 'nearest',
  };

  element.scrollIntoView({ ...defaultOptions, ...options });
};

/**
 * Intersection Observer for scroll animations
 */
export class ScrollAnimationObserver {
  private observer: IntersectionObserver;
  private elements: Map<Element, string> = new Map();

  constructor(
    threshold: number = 0.1,
    rootMargin: string = '0px 0px -50px 0px'
  ) {
    this.observer = new IntersectionObserver(
      this.handleIntersection.bind(this),
      { threshold, rootMargin }
    );
  }

  private handleIntersection(entries: IntersectionObserverEntry[]) {
    entries.forEach(entry => {
      const animationClass = this.elements.get(entry.target);
      if (!animationClass) return;

      if (entry.isIntersecting) {
        entry.target.classList.add(animationClass);
      } else {
        entry.target.classList.remove(animationClass);
      }
    });
  }

  observe(element: Element, animationClass: string) {
    this.elements.set(element, animationClass);
    this.observer.observe(element);
  }

  unobserve(element: Element) {
    this.elements.delete(element);
    this.observer.unobserve(element);
  }

  disconnect() {
    this.observer.disconnect();
    this.elements.clear();
  }
}

/**
 * Stagger animation utility for lists
 */
export const staggerChildren = (
  container: HTMLElement,
  animationClass: string,
  delay: number = 100
) => {
  if (prefersReducedMotion()) {
    // Apply animation immediately without stagger
    container.querySelectorAll(':scope > *').forEach(child => {
      child.classList.add(animationClass);
    });
    return;
  }

  container.querySelectorAll(':scope > *').forEach((child, index) => {
    setTimeout(() => {
      child.classList.add(animationClass);
    }, index * delay);
  });
};

/**
 * Typing animation utility
 */
export class TypingAnimation {
  private element: HTMLElement;
  private text: string;
  private speed: number;
  private currentIndex: number = 0;
  private animationId: number | null = null;

  constructor(element: HTMLElement, text: string, speed: number = 50) {
    this.element = element;
    this.text = text;
    this.speed = prefersReducedMotion() ? 0 : speed;
  }

  start(): Promise<void> {
    return new Promise(resolve => {
      if (this.speed === 0) {
        // No animation for reduced motion
        this.element.textContent = this.text;
        resolve();
        return;
      }

      const animate = () => {
        if (this.currentIndex < this.text.length) {
          this.element.textContent = this.text.slice(0, this.currentIndex + 1);
          this.currentIndex++;
          this.animationId = window.setTimeout(animate, this.speed);
        } else {
          resolve();
        }
      };

      animate();
    });
  }

  stop() {
    if (this.animationId) {
      window.clearTimeout(this.animationId);
      this.animationId = null;
    }
    this.element.textContent = this.text;
  }
}

/**
 * Parallax scroll utility (respects reduced motion)
 */
export const createParallaxEffect = (
  element: HTMLElement,
  speed: number = 0.5
) => {
  if (prefersReducedMotion()) {
    return () => {}; // No-op for reduced motion
  }

  const handleScroll = () => {
    const scrolled = window.pageYOffset;
    const parallax = scrolled * speed;
    element.style.transform = `translateY(${parallax}px)`;
  };

  window.addEventListener('scroll', handleScroll, { passive: true });

  return () => {
    window.removeEventListener('scroll', handleScroll);
  };
};

/**
 * Hover animation utilities
 */
export const hoverAnimations = {
  scale: (element: HTMLElement, scale: number = 1.05) => {
    if (prefersReducedMotion()) return;

    const handleMouseEnter = () => {
      element.style.transform = `scale(${scale})`;
    };

    const handleMouseLeave = () => {
      element.style.transform = 'scale(1)';
    };

    element.addEventListener('mouseenter', handleMouseEnter);
    element.addEventListener('mouseleave', handleMouseLeave);

    return () => {
      element.removeEventListener('mouseenter', handleMouseEnter);
      element.removeEventListener('mouseleave', handleMouseLeave);
    };
  },

  glow: (element: HTMLElement, color: string = 'rgba(255, 255, 255, 0.1)') => {
    if (prefersReducedMotion()) return;

    const handleMouseEnter = () => {
      element.style.boxShadow = `0 0 20px ${color}`;
    };

    const handleMouseLeave = () => {
      element.style.boxShadow = 'none';
    };

    element.addEventListener('mouseenter', handleMouseEnter);
    element.addEventListener('mouseleave', handleMouseLeave);

    return () => {
      element.removeEventListener('mouseenter', handleMouseEnter);
      element.removeEventListener('mouseleave', handleMouseLeave);
    };
  },
};
